import json

submission = []

for sequence_id in range(1, 5121):
    for frame in range(1, 6):
        submission.append(
            {"sequence_id": sequence_id, "frame": frame, "num_objects": 0, "object_coords": []}
        )

with open("my_submission.json", "w") as outfile:
    json.dump(submission, outfile)
