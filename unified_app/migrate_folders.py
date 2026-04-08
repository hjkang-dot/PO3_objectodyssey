import os
import shutil
import json
from pathlib import Path

def migrate():
    root_dir = Path(__file__).parent
    save_dir = root_dir / "saved_stories"
    
    if not save_dir.exists():
        print("No saved_stories directory found. Skipping.")
        return

    # saved_stories 폴더 내의 모든 .json 파일 찾기 (하위 폴더 제외)
    json_files = [f for f in save_dir.iterdir() if f.is_file() and f.suffix == ".json"]
    
    if not json_files:
        print("No legacy JSON files to migrate.")
        return

    print(f"Found {len(json_files)} files to migrate.")

    for file_path in json_files:
        # 파일명에서 확장자 제거한 이름으로 폴더명 생성
        folder_name = file_path.stem
        target_folder = save_dir / folder_name
        
        try:
            # 폴더 생성
            target_folder.mkdir(exist_ok=True)
            
            # 파일을 폴더 안으로 story.json 이라는 이름으로 이동
            target_path = target_folder / "story.json"
            shutil.move(str(file_path), str(target_path))
            
            print(f"Migrated: {file_path.name} -> {folder_name}/story.json")
        except Exception as e:
            print(f"Failed to migrate {file_path.name}: {e}")

if __name__ == "__main__":
    migrate()
