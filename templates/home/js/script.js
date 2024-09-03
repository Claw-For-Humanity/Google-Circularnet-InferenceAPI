window.onload = function() {
    setTimeout(function() {
        // Add the hidden class to start the fade-out
        document.getElementById('overlay').classList.add('hidden');

        // After the transition, remove the overlay from the DOM
        setTimeout(function() {
            document.getElementById('overlay').style.display = 'none';
            document.body.style.overflow = 'auto'; // Restores scrolling
        }, 2000); // This duration should match the CSS transition duration
    }, 2000); // Time before the overlay starts fading out
};
