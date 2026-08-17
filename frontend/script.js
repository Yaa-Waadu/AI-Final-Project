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

const practiceLetter =
    document.getElementById("practiceLetter");

const feedbackBox =
    document.getElementById("feedbackBox");

const feedbackText =
    document.getElementById("feedbackText");





const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

alphabet.split("").forEach(letter => {

    const card = document.createElement("div");

    card.className = "letter-card";

    card.textContent = letter;

    alphabetGrid.appendChild(card);

});


alphabet.split("").forEach(letter => {

    const option = document.createElement("option");

    option.value = letter;

    option.textContent = letter;

    practiceLetter.appendChild(option);

});




uploadArea.addEventListener("click", () => {

    imageInput.click();

});




imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (file) {

        displayImage(file);

    }

});




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




removeButton.addEventListener("click", (event) => {

    event.stopPropagation();

    imageInput.value = "";

    preview.src = "";

    previewContainer.classList.remove("active");

    uploadContent.style.display = "block";

    predictBtn.disabled = true;

    resultCard.classList.remove("active");

    feedbackBox.classList.remove("active");

    confidenceProgress.style.width = "0%";

});





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


    if (!file) {

        return;

    }


    if (!file.type.startsWith("image/")) {

        alert("Please select an image file.");

        return;

    }


    // Add the dropped file to the file input
    const dataTransfer = new DataTransfer();

    dataTransfer.items.add(file);

    imageInput.files = dataTransfer.files;


    // Display the dropped image
    displayImage(file);

});



predictBtn.addEventListener("click", async () => {

    const file = imageInput.files[0];

    if (!file) {

        return;

    }


    loading.classList.add("active");

    resultCard.classList.remove("active");

    predictBtn.disabled = true;


    // Create form data for the Flask backend

    const formData = new FormData();

    formData.append("image", file);

    if (practiceLetter.value) {

        formData.append("letter", practiceLetter.value);

    }


    try {

        // Send image to the Python backend

        const response = await fetch("/predict", {

            method: "POST",

            body: formData

        });


        // Convert response to JSON

        const data = await response.json();


        // Check for backend errors

        if (!response.ok) {

            throw new Error(
                data.error || "Prediction failed."
            );

        }


        // Get prediction from the model

        const predictedLetter =
            data.prediction.toUpperCase();


        const predictedConfidence =
            data.confidence;


        // Display prediction

        prediction.textContent =
            predictedLetter;


        // Display confidence

        confidence.textContent =
            `${predictedConfidence}%`;


        // Display coaching feedback, if the backend generated any

        if (data.feedback) {

            feedbackText.textContent = data.feedback;

            feedbackBox.classList.add("active");

        } else {

            feedbackBox.classList.remove("active");

        }


        // Show result card

        resultCard.classList.add("active");


        // Animate confidence bar

        setTimeout(() => {

            confidenceProgress.style.width =
                `${predictedConfidence}%`;

        }, 100);


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        alert(
            "Unable to get a prediction. Please make sure the backend is running."
        );

    } finally {

        loading.classList.remove("active");

        predictBtn.disabled = false;

    }

});