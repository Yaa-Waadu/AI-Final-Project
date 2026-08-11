const imageInput = document.getElementById("imageInput");

const uploadArea = document.getElementById("uploadArea");

const uploadContent = document.getElementById("uploadContent");

const previewContainer =
    document.getElementById("previewContainer");

const preview =
    document.getElementById("preview");

const removeButton =
    document.getElementById("removeButton");

const predictBtn =
    document.getElementById("predictBtn");

const loading =
    document.getElementById("loading");

const resultCard =
    document.getElementById("resultCard");

const prediction =
    document.getElementById("prediction");

const confidence =
    document.getElementById("confidence");

const confidenceProgress =
    document.getElementById("confidenceProgress");

const alphabetGrid =
    document.getElementById("alphabetGrid");



/* =========================
   CREATE ALPHABET
========================= */

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

alphabet.split("").forEach(letter => {

    const card = document.createElement("div");

    card.className = "letter-card";

    card.textContent = letter;

    alphabetGrid.appendChild(card);

});



/* =========================
   OPEN FILE SELECTOR
========================= */

uploadArea.addEventListener("click", () => {

    imageInput.click();

});



/* =========================
   IMAGE SELECTION
========================= */

imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (file) {

        displayImage(file);

    }

});



/* =========================
   DISPLAY IMAGE
========================= */

function displayImage(file) {

    if (!file.type.startsWith("image/")) {

        alert("Please select an image file.");

        return;

    }


    const reader = new FileReader();


    reader.onload = (event) => {

        preview.src = event.target.result;

        previewContainer.classList.add("active");

        uploadContent.style.display = "none";

        predictBtn.disabled = false;

        resultCard.classList.remove("active");

    };


    reader.readAsDataURL(file);

}



/* =========================
   REMOVE IMAGE
========================= */

removeButton.addEventListener("click", (event) => {

    event.stopPropagation();

    imageInput.value = "";

    preview.src = "";

    previewContainer.classList.remove("active");

    uploadContent.style.display = "block";

    predictBtn.disabled = true;

    resultCard.classList.remove("active");

});



/* =========================
   DRAG AND DROP
========================= */

uploadArea.addEventListener("dragover", (event) => {

    event.preventDefault();

    uploadArea.classList.add("dragging");

});


uploadArea.addEventListener("dragleave", () => {

    uploadArea.classList.remove("dragging");

});


uploadArea.addEventListener("drop", (event) => {

    event.preventDefault();

    uploadArea.classList.remove("dragging");


    const file = event.dataTransfer.files[0];

    if (file) {

        displayImage(file);

    }

});



/* =========================
   PREDICT BUTTON
========================= */

predictBtn.addEventListener("click", () => {

    if (!imageInput.files[0]) {

        return;

    }


    loading.classList.add("active");

    resultCard.classList.remove("active");

    predictBtn.disabled = true;


    /*
        TEMPORARY DEMO

        This will later be replaced with
        a request to our Python backend.

        Example:

        fetch("http://localhost:5000/predict", {
            method: "POST",
            body: formData
        })
    */


    setTimeout(() => {

        loading.classList.remove("active");

        predictBtn.disabled = false;


        // Temporary demonstration result
        const demoLetter = "A";

        const demoConfidence = 98.6;


        prediction.textContent = demoLetter;

        confidence.textContent =
            `${demoConfidence}%`;


        resultCard.classList.add("active");


        setTimeout(() => {

            confidenceProgress.style.width =
                `${demoConfidence}%`;

        }, 100);


    }, 1500);

});