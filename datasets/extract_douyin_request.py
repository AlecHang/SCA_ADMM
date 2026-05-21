import csv

def extract_video_id(input_file, output_file):
    video_ids = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        video_id_index = header.index('video_id')
        
        for row in reader:
            if len(row) > video_id_index:
                video_ids.append(row[video_id_index])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for video_id in video_ids:
            f.write(video_id + '\n')
    
    print(f"Successfully extracted {len(video_ids)} video_ids to {output_file}")

if __name__ == '__main__':
    input_file = 'dy_action.csv'
    output_file = 'douyin-request.txt'
    extract_video_id(input_file, output_file)