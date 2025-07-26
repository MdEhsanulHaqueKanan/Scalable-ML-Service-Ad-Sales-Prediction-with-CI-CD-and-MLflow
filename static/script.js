// static/script.js

// Wait for the entire HTML document to be loaded before running the script
document.addEventListener('DOMContentLoaded', () => {
    
    // Get references to the HTML elements we need to interact with
    const form = document.getElementById('prediction-form');
    const resultText = document.getElementById('result-text');
    const submitButton = document.getElementById('submit-button');

    // Listen for the 'submit' event on the form
    form.addEventListener('submit', async (event) => {
        // Prevent the default form submission behavior (which would reload the page)
        event.preventDefault();

        // Change button text and disable it to prevent multiple clicks
        submitButton.textContent = 'Predicting...';
        submitButton.disabled = true;
        resultText.textContent = '---';

        // Create a FormData object from the form to easily access its data
        const formData = new FormData(form);
        // Convert the form data into a plain JavaScript object
        const data = Object.fromEntries(formData.entries());

        try {
            // Send a POST request to our /predict endpoint using the Fetch API
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                // Convert the JavaScript object to a JSON string for the request body
                body: JSON.stringify(data)
            });

            // Parse the JSON response from the server
            const result = await response.json();

            // Check if the response was successful (HTTP status 200-299)
            if (response.ok) {
                // Display the prediction in a user-friendly format
                resultText.textContent = `$${result.sale_amount_prediction.toFixed(2)}`;
            } else {
                // Display a clear error message if something went wrong on the server
                resultText.textContent = `Error: ${result.error}`;
            }

        } catch (error) {
            // Handle network errors (e.g., server is down)
            resultText.textContent = 'An error occurred. Please try again.';
            console.error('Fetch error:', error);
        } finally {
            // Re-enable the button and reset its text, regardless of success or failure
            submitButton.textContent = 'Predict Sale Amount';
            submitButton.disabled = false;
        }
    });
});