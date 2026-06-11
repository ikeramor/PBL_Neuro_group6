import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check myelin map generation logs for errors.")
    parser.add_argument("--log_dir", type=str, default="/home/jovyan/Desktop/PBL_Neuro/myelin_maps",
                        help="Path to the directory containing the _log.txt files.")
    parser.add_argument("--error_text", type=str, default="ERROR - Image creation failed.",
                        help="The text inside the log that indicates a failure.")
    
    args = parser.parse_args()

    # Pattern to find all log files
    log_pattern = os.path.join(args.log_dir, "*_log.txt")
    log_files = glob.glob(log_pattern)

    if not log_files:
        print(f"No log files found in {args.log_dir}. Please check the path.")
        return

    failed_subjects = []

    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if args.error_text in content:
                    # Extract the subject ID from the filename (e.g., sub-01_log.txt)
                    filename = os.path.basename(log_file)
                    subject_id = filename.replace("_log.txt", "")
                    failed_subjects.append(subject_id)
        except Exception as e:
            print(f"Could not read {log_file}: {e}")

    # Print the results
    print("-" * 50)
    print(f"Total logs analyzed: {len(log_files)}")
    print(f"Total failures detected: {len(failed_subjects)}")
    print("-" * 50)

    if failed_subjects:
        print("Subjects that failed to generate the myelin map:")
        for sub in failed_subjects:
            print(f" - {sub}")
    else:
        print("All subjects processed successfully! No errors found in the logs.")
    print("-" * 50)

if __name__ == "__main__":
    main()
