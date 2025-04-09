import random
import concurrent.futures

from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.vanilla import LabeledFewShot


class BootstrapFewShotWithRandomSearchFast(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_settings={},
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=6,
        max_errors=10,
        stop_at_score=None,
        metric_threshold=None,
        num_workers=20
    ):
        self.metric = metric
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds

        self.num_threads = num_threads  # used in Evaluate
        self.num_workers = num_workers # used to bootstrap in parallel
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

        print(f"Going to sample between {self.min_num_samples} and {self.max_num_samples} traces per predictor.")
        print(f"Will attempt to bootstrap {self.num_candidate_sets} candidate sets.")

    def compile(
        self,
        student,
        *,
        teacher=None,
        trainset,
        valset=None,
        restrict=None,
        labeled_sample=True,
    ):
        self.trainset = trainset
        self.valset = valset or trainset  # TODO: FIXME: Note this choice.

        # Create a list of seeds to process (respecting any "restrict" filter)
        seeds = [
            seed for seed in range(-3, self.num_candidate_sets)
            if restrict is None or seed in restrict
        ]

        # Define the work to be done for each seed.
        # This function mimics exactly what you had in the loop.
        def process_seed(seed):
            trainset_copy = list(self.trainset)

            if seed == -3:
                # zero-shot
                program = student.reset_copy()

            elif seed == -2:
                # labels only
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program = teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)

            elif seed == -1:
                # unshuffled few-shot
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=self.max_errors,
                )
                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            else:
                assert seed >= 0, seed

                # Shuffle and choose a random sample size deterministically with the seed.
                random.Random(seed).shuffle(trainset_copy)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=self.max_errors,
                )
                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            evaluate = Evaluate(
                devset=self.valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=self.max_errors,
                display_table=False,
                display_progress=True,
            )

            score, subscores = evaluate(program, return_all_scores=True)

            # Assertion-aware Optimization
            if hasattr(program, "_suggest_failures"):
                score = score - program._suggest_failures * 0.2
            if hasattr(program, "_assert_failures"):
                score = 0 if program._assert_failures > 0 else score

            return {"score": score, "subscores": subscores, "seed": seed, "program": program}

        # Process all candidate seeds concurrently.
        # Using executor.map ensures the results come back in the same order as the seeds list.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            candidate_results = list(executor.map(process_seed, seeds))

        scores = []
        score_data = []
        best_program = None

        # Process the results in sequence (as if they had been computed sequentially).
        for result in candidate_results:
            score = result["score"]
            seed = result["seed"]
            program = result["program"]
            subscores = result["subscores"]

            if len(scores) == 0 or score > max(scores):
                print("New best score:", score, "for seed", seed)
                best_program = program
            scores.append(score)
            print(f"Scores so far: {scores}")
            print(f"Best score so far: {max(scores)}")

            score_data.append({
                "score": score,
                "subscores": subscores,
                "seed": seed,
                "program": program
            })

            if self.stop_at_score is not None and score >= self.stop_at_score:
                print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
                break

        # Attach candidate programs sorted by score (highest first) to best_program.
        best_program.candidate_programs = sorted(score_data, key=lambda x: x["score"], reverse=True)
        print(f"{len(best_program.candidate_programs)} candidate programs found.")

        return best_program