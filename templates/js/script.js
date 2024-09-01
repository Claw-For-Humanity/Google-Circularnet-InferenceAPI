async function uploadImage(event) {
    event.preventDefault();
    const submitButton = document.getElementById('submit-button');
    const resultDiv = document.getElementById('result');
    
    // Disable the submit button
    submitButton.disabled = true;
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
        
        if (response.ok) {
            const data = await response.json();
            const filename = data.filename;

            // Display the image directly on the page
            resultDiv.innerHTML = `
                <h2>Processing Complete</h2>
                <img src="/${filename}" alt="Processed Image" class="img-fluid" />
                <button onclick="window.location.reload();" class="btn btn-primary mt-2">Reset</button>
            `;
        } else {
            console.error('Upload failed');
            resultDiv.innerHTML = `<p class="text-danger">An error occurred. Please try again.</p>`;
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.innerHTML = `<p class="text-danger">An error occurred. Please try again.</p>`;
    } finally {
        // Re-enable the submit button after processing
        submitButton.disabled = false;
        submitButton.textContent = 'Upload and Process';
    }
}



// var count = 1
//         setTimeout(demo, 500)
//         setTimeout(demo, 700)
//         setTimeout(demo, 900)
//         setTimeout(reset, 2000)

//         setTimeout(demo, 2500)
//         setTimeout(demo, 2750)
//         setTimeout(demo, 3050)

//         var mousein = false
//         function demo() {
//             if(mousein) return
//             document.getElementById('demo' + count++)
//                 .classList.toggle('hover')
//         }

//         function demo2() {
//             if(mousein) return
//             document.getElementById('demo2')
//                 .classList.toggle('hover')
//         }

//         function reset() {
//             count = 1
//             var hovers = document.querySelectorAll('.hover')
//             for(var i = 0; i < hovers.length; i++ ) {
//                 hovers[i].classList.remove('hover')
//             }
//         }

//         document.addEventListener('mouseover', function() {
//             mousein = true
//             reset()
//         })