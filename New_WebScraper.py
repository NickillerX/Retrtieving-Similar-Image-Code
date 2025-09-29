from icrawler.builtin import GoogleImageCrawler
import os
import time
def download_exact_images(search_term, output_dir, target_num=50, batch_size=5):
    os.makedirs(output_dir, exist_ok=True)
    # Count already downloaded images
    def count_images():
        return len([
            f for f in os.listdir(output_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
    downloaded = count_images()
    attempts = 0
    while downloaded < target_num:
        remaining = target_num - downloaded
        print(f" {search_term}: Downloading next {min(batch_size, remaining)} images (currently have {downloaded})")
        crawler = GoogleImageCrawler(storage={"root_dir": output_dir})
        crawler.crawl(
            keyword=search_term,
            max_num=min(batch_size, remaining),
            min_size=(224, 224),
            file_idx_offset='auto'
        )
        time.sleep(4)  # pause to avoid rate-limiting
        downloaded = count_images()
        attempts += 1
        if attempts > 10:
            print(f" Stopping early: Too many retries for {search_term}")
            break
    print(f" {search_term}: Total downloaded = {downloaded} images")
classes = [
            'Golden Retriever puppy',
            'German Shepherd police dog',
            'dog running in a park',
            'French Bulldog portrait',
            'poodle in show'
          ]
base_dir = 'C:/Users/Nick/Desktop/FinalDataset/animals cure/more dog'
for cls in classes:
    class_dir = os.path.join(base_dir, cls)
    print(f"\n Starting download for class: {cls}")
    download_exact_images(cls, class_dir, target_num=20,batch_size=20)