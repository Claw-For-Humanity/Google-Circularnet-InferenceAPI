
async function uploadImage(event) {
    event.preventDefault();
    const submitButton = document.getElementById('submit-button');
    const resultDiv = document.getElementById('result');
    
    // Disable the submit button
    submitButton.hidden = true;
    submitButton.textContent = 'Processing...';

    const formData = new FormData(document.getElementById('upload-form'));

    try {
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
    
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (response.ok) {
            const data = await response.json();
            const filename = data.filename;
    
            // Display the image directly on the page
            resultDiv.innerHTML = `
                <h3>Processing Complete</h3>
                <img src="/${filename}" alt="Processed Image" class="img-fluid" />
                <button onclick="window.location.reload();" class="btn button-gradient mt-2">Restart!</button>
            `;
        } 
        else if (response.status === 400) {
            console.error('Upload failed: format has to be .png, .jpg,or .jpeg', response.statusText);
            resultDiv.innerHTML = `<p class="text-danger">Wrong format. Please upload a valid file.</p>`;
        }
        else {
            console.error('Upload failed', response.statusText);
            resultDiv.innerHTML = `<p class="text-danger">An error occurred. Please try again.</p>`;
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.innerHTML = `<p class="text-danger">An error occurred. Please try again.</p>`;
    }
    
}

