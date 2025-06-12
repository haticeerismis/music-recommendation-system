🎵 Müzik Öneri Sistemi

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

```bash
git clone https://github.com/haticeerismis/Muzik-Oneri-Sistemi.git
cd Muzik-Oneri-Sistemi
pip install -r requirements.txt
Spotify API bilgilerinizi .env dosyası içinde tanımlayın:
SPOTIFY_CLIENT_ID=xxx
SPOTIFY_CLIENT_SECRET=xxx
python app.py


```
