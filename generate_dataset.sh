python generate_dataset/crawl_metadata.py --data data
python generate_dataset/update_metadata.py --data data
python generate_dataset/detect_missing_metadata.py --data data
python generate_dataset/export_dataset_to_csv.py --data data --output .