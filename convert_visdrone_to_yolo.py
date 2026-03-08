import os
import cv2
from tqdm import tqdm
import shutil

# AYARLAR
# VisDrone sınıfları: 0:Ignore, 1:Pedestrian, 2:People, 3:Bicycle, 4:Car, 5:Van, 6:Truck, 7:Tricycle, 8:Awning-tricycle, 9:Bus, 10:Motor, 11:Others
# Biz sadece 1 ve 2'yi alıp bunları tek bir sınıf (0: Human) yapacağız.
TARGET_CLASSES = [1, 2] 

def convert_to_yolo(size, box):
    # VisDrone: x_min, y_min, width, height
    # YOLO: x_center, y_center, width, height (Hepsi 0-1 arasında normalize edilmiş)
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

def process_folder(folder_name, output_folder):
    input_images = f"{folder_name}/images"
    input_labels = f"{folder_name}/annotations"
    
    # Yeni klasörleri oluştur
    output_images_dir = os.path.join(output_folder, 'images')
    output_labels_dir = os.path.join(output_folder, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    file_list = [f for f in os.listdir(input_images) if f.endswith('.jpg')]

    print(f"İşleniyor: {folder_name} -> {len(file_list)} resim bulundu.")

    for filename in tqdm(file_list):
        img_id = filename.split('.')[0]
        
        # Resmi oku (Boyutları almak için)
        img_path = os.path.join(input_images, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        height, width, _ = img.shape

        # Label dosyasını oku
        txt_path = os.path.join(input_labels, f"{img_id}.txt")
        if not os.path.exists(txt_path): continue

        yolo_lines = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(',')
                class_id = int(data[5])
                
                # Sadece insanları al (Sınıf 1 ve 2)
                if class_id in TARGET_CLASSES:
                    box = (float(data[0]), float(data[1]), float(data[2]), float(data[3]))
                    # Koordinatları dönüştür
                    yolo_box = convert_to_yolo((width, height), box)
                    # YOLO dosya satırı: class_id x y w h (Bizim class_id hep 0 olacak = INSAN)
                    yolo_lines.append(f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")

        # Eğer resimde hiç insan yoksa o resmi ve boş txt dosyasını da eğitime katabiliriz (Background image)
        # Ama şimdilik sadece insan olanları kaydedelim ki eğitim hızlı olsun.
        if len(yolo_lines) > 0:
            # Resmi kopyala
            shutil.copy(img_path, os.path.join(output_images_dir, filename))
            # Label dosyasını yaz
            with open(os.path.join(output_labels_dir, f"{img_id}.txt"), 'w') as f:
                f.write('\n'.join(yolo_lines))

# BURAYI KENDİ KLASÖR İSİMLERİNE GÖRE DÜZENLE
if __name__ == "__main__":
    # Eğitim setini dönüştür
    process_folder("VisDrone2019-DET-train", "datasets/visdrone_human/train")
    # Test/Val setini dönüştür
    process_folder("VisDrone2019-DET-val", "datasets/visdrone_human/val")
    print("Dönüştürme tamamlandı! 'datasets' klasörünü kontrol et.")