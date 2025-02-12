document.addEventListener("DOMContentLoaded", function () {
    const menuClassification = document.getElementById("menu-classification");
    const menuCnnModel = document.getElementById("menu-cnn-model");
    const classificationSection = document.getElementById("classification-section");
    const cnnModelSection = document.getElementById("cnn-model-section");
    const imageUpload = document.getElementById("image-upload");
    const submitBtn = document.getElementById("submit-btn");
    const clearBtn = document.getElementById("clear-btn");
    const predictionResult = document.getElementById("prediction-result");
    const uploadedImage = document.getElementById("uploaded-image");

    // Menambahkan class active untuk menu Image Classification saat halaman pertama kali dimuat
    classificationSection.style.display = "block";
    cnnModelSection.style.display = "none";
    menuClassification.classList.add("active");
    menuCnnModel.classList.remove("active");

    // Navigasi sidebar
    menuClassification.addEventListener("click", function () {
        classificationSection.style.display = "block";
        cnnModelSection.style.display = "none";
        menuClassification.classList.add("active");
        menuCnnModel.classList.remove("active");
    });

    menuCnnModel.addEventListener("click", function () {
        classificationSection.style.display = "none";
        cnnModelSection.style.display = "block";
        menuCnnModel.classList.add("active");
        menuClassification.classList.remove("active");
    });

    // Fungsi untuk tab dalam CNN Model
    window.openTab = function (evt, tabName) {
        let tabcontent = document.getElementsByClassName("tabcontent");
        for (let i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }

        let tablinks = document.getElementsByClassName("tablinks");
        for (let i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }

        // Menampilkan tab yang relevan
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";

        // Jika tab ROC Curve dibuka, pastikan gambar ROC curve dimuat
        if (tabName === 'ROCcurve') {
            loadROCCurveImages();
        }
    };

    // Fungsi untuk mengunggah gambar dan mendapatkan prediksi
    submitBtn.addEventListener("click", function () {
        let file = imageUpload.files[0];
        if (!file) {
            alert("Pilih gambar terlebih dahulu!");
            return;
        }

        let formData = new FormData();
        formData.append("file", file);

        fetch("/submit", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                predictionResult.innerHTML = `<p style="color:red;">${data.message}</p>`;
                predictionResult.style.display = "block";
            } else {
                // Menampilkan gambar yang diunggah
                uploadedImage.src = data.img_path; // Tampilkan gambar yang diunggah
                uploadedImage.style.display = "block"; // Pastikan gambar ditampilkan

                // Tampilkan hasil prediksi setelah gambar
                predictionResult.innerHTML = `
                    <div class="prediction-text">
                        <p><strong> Prediksi NasNet:</strong> ${data.predictionnasnet} (${data.confidencenasnet})</p>
                        <p><strong> Prediksi VGG16:</strong> ${data.predictionvgg} (${data.confidencvgg})</p>
                        <p><strong> Prediksi Xception:</strong> ${data.predictionxception} (${data.confidencexception})</p>
                        <p><strong> Prediksi CNN:</strong> ${data.predictioncnn} (${data.confidencecnn})</p>
                    </div>
                `;
                predictionResult.style.display = "block";
            }
        })
        .catch(error => console.error("Error:", error));
    });

    // Tombol Clear untuk mereset form dan tampilan
    clearBtn.addEventListener("click", function () {
        imageUpload.value = "";
        uploadedImage.style.display = "none"; // Menyembunyikan gambar yang diunggah
        predictionResult.innerHTML = "";
    });

    // Media query untuk mengubah sidebar menjadi ikon saat ukuran layar lebih kecil
    const mediaQuery = window.matchMedia("(max-width: 768px)");

    // Mengubah sidebar menjadi ikon saat layar lebih kecil dari 768px
    function updateSidebar() {
        const sidebar = document.querySelector(".sidebar");
        if (mediaQuery.matches) {
            sidebar.classList.add("icon-mode");
        } else {
            sidebar.classList.remove("icon-mode");
        }
    }

    // Memastikan sidebar berubah saat pertama kali halaman dimuat
    updateSidebar();

    // Mengupdate sidebar saat ukuran layar berubah
    mediaQuery.addEventListener("change", updateSidebar);

    // Fungsi untuk memuat gambar ROC Curve
    function loadROCCurveImages() {
        console.log('Loading ROC Curve images...');
        // Menampilkan ROC Curve di tab yang relevan
        document.getElementById("roc_curve_image").src = "static/images/roc_curve.png"; // Ubah sesuai model
    }
});