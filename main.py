import json
import datetime


def create_submission_file():
    """Write the submission file"""

    submission = []

    for sequence_id in range(1, 5121):
        for frame in range(1, 6):
            submission.append(
                {
                    "sequence_id": sequence_id,
                    "frame": frame,
                    "num_objects": 0,
                    "object_coords": [],
                }
            )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%m-%S")

    with open(f"sub_{timestamp}.json", "w") as outfile:
        json.dump(submission, outfile)


def main(): ...


if __name__ == "__main__":
    main()
