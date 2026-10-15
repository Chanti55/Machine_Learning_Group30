
# Usage:
# python submission_tools/record_score.py version_21 0.7947 or simply: python submission_tools/record_score.py

import sys
from pathlib import Path

# Ensure submission_tools directory is on Python path
tools_dir = Path(__file__).resolve().parent
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from submission_tracker import record_kaggle_score, update_leaderboard_file, list_submission_versions

def main():
    if len(sys.argv) >= 3:
        vname = sys.argv[1].strip()
        try:
            score = float(sys.argv[2])
        except ValueError:
            print(f"Error: Score '{sys.argv[2]}' is not a valid float.")
            sys.exit(1)
        private_score = float(sys.argv[3]) if len(sys.argv) >= 4 else None
    elif len(sys.argv) == 2 and sys.argv[1] in ["--list", "-l", "list"]:
        df = list_submission_versions()
        print("\n--- Current Submissions Leaderboard ---")
        print(df.to_string())
        return
    else:
        print("\n🏆 Kaggle Score Recorder for Group 30")
        print("--------------------------------------")
        vname = input("Enter version name (e.g. version_21): ").strip()
        score_str = input("Enter public leaderboard score (e.g. 0.7947): ").strip()
        try:
            score = float(score_str)
        except ValueError:
            print(f"Error: '{score_str}' is not a valid score.")
            sys.exit(1)
        private_score = None

    record_kaggle_score(vname, score, private_score=private_score)
    print("\n The following files have been updated:")
    print(f"   - submissions/{vname}/cv_metrics.json")
    print(f"   - submissions/{vname}/REPORT.md")
    print("   - submissions/LEADERBOARD.md\n")

if __name__ == "__main__":
    main()
