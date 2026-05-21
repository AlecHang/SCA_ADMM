def process_netflix_qualifying(input_file, output_file):
    video_access = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        current_video_id = None
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith(':'):
                current_video_id = line[:-1]
            else:
                parts = line.split(',')
                if len(parts) >= 2 and current_video_id:
                    user_id = parts[0]
                    access_time = parts[1]
                    video_access.append((access_time, current_video_id))
    
    video_access.sort(key=lambda x: x[0])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for access_time, video_id in video_access:
            f.write(video_id + '\n')
    
    print(f"Successfully processed {len(video_access)} access records")
    print(f"Output saved to {output_file}")

if __name__ == '__main__':
    input_file = 'netflix-qualifying.txt'
    output_file = 'netflix-request.txt'
    process_netflix_qualifying(input_file, output_file)