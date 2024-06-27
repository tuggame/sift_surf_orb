import os

import pandas as pd
import requests

# TSV dosyasını oku
tsv_file_path = 'dataset_statics/photos.tsv'
df = pd.read_csv(tsv_file_path, sep='\t')

url_column_name = 'photo_image_url'

output_folder = 'dataset_images'
os.makedirs(output_folder, exist_ok=True)

# URL'leri al ve görselleri indir
for index, row in df.iterrows():
    img_url = row[url_column_name]
    img_name = os.path.join(output_folder, f'image_{index}.jpg')

    try:
        response = requests.get(img_url)
        if response.status_code == 200:
            with open(img_name, 'wb') as f:
                f.write(response.content)
            print(f'{img_name} başarıyla indirildi.')
        else:
            print(f'{img_url} indirilemedi, status kodu: {response.status_code}')
    except Exception as e:
        print(f'{img_url} indirilemedi, hata: {str(e)}')

print('İndirme işlemi tamamlandı.')
