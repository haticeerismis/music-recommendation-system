🎵 Müzik Öneri Sistemi
ARAYÜZ 
![image](https://github.com/user-attachments/assets/3fc5774a-9034-468d-8e5e-70347bd6d96e)
![image](https://github.com/user-attachments/assets/c60d413f-a6a2-4cc2-ad63-798af1347b29)
![image](https://github.com/user-attachments/assets/85070e55-9eff-4320-9687-98d1a9c30dd2)
![image](https://github.com/user-attachments/assets/57b37c81-88aa-4700-b78c-582be386e6ba)

Bu proje, kullanıcıların ruh haline (mood), keşfetme isteğine ya da seçtiği bir şarkıya göre benzer parçalarla öneri almasını sağlayan bir müzik öneri sistemidir. Flask ile geliştirilmiş backend yapısı, kullanıcı dostu bir web arayüzü ile desteklenmiştir.

---

🚀 Özellikler

🎧 **Moda Göre Öneri**: Mutlu, üzgün, enerjik veya sakin hissettiğinize göre şarkılar önerilir.
🔁 **Benzer Şarkı Önerisi**: Seçtiğiniz bir şarkıya en çok benzeyen 5 farklı parça önerilir.
✨ **Keşfetme Modu**: Enerjik ve dans edilebilir şarkılardan rastgele öneriler yapılır.
📺 **Spotify Player Desteği**: Önerilen şarkılar embed player ile sayfa üzerinde dinlenebilir.
🔍 **Dropdown Menü ile Arama**: Şarkılar isimleriyle listelenir, kullanıcı istediği parçayı kolayca seçebilir.

---

## 🛠️ Kullanılan Teknolojiler

- Python + Flask
- HTML, CSS, JavaScript
- scikit-learn
- pandas
- Spotify API

---

## 💾 Kurulum

> Projeyi kendi bilgisayarınızda çalıştırmak için:
> DataSet proje ana dizinine eklenmeli. Veriseti Link: https://drive.google.com/drive/folders/1y7OTYaZLMF7FoCTvvRHSm2O5C-p7_yPj?usp=sharing

```bash
git clone https://github.com/haticeerismis/Muzik-Oneri-Sistemi.git
cd Muzik-Oneri-Sistemi
pip install -r requirements.txt
Spotify API bilgilerinizi .env dosyası içinde tanımlayın:
SPOTIFY_CLIENT_ID=xxx
SPOTIFY_CLIENT_SECRET=xxx
python app.py


```
