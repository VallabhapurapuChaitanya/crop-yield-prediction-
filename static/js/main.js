// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeButton = alert.querySelector('.btn-close');
            if (closeButton) {
                closeButton.click();
            }
        }, 5000);
    });

    // Password strength indicator for registration
    const passwordInput = document.getElementById('password');
    if (passwordInput) {
        passwordInput.addEventListener('input', function() {
            const password = this.value;
            let strength = 0;

            // Check password length
            if (password.length >= 8) strength += 1;

            // Check for mixed case
            if (password.match(/[a-z]/) && password.match(/[A-Z]/)) strength += 1;

            // Check for numbers
            if (password.match(/\d/)) strength += 1;

            // Check for special characters
            if (password.match(/[^a-zA-Z\d]/)) strength += 1;

            // Update strength indicator
            let strengthText = '';
            let strengthClass = '';

            switch(strength) {
                case 0:
                case 1:
                    strengthText = 'Weak';
                    strengthClass = 'text-danger';
                    break;
                case 2:
                    strengthText = 'Moderate';
                    strengthClass = 'text-warning';
                    break;
                case 3:
                    strengthText = 'Good';
                    strengthClass = 'text-primary';
                    break;
                case 4:
                    strengthText = 'Strong';
                    strengthClass = 'text-success';
                    break;
            }

            // Create or update the strength indicator
            let strengthIndicator = document.getElementById('password-strength');
            if (!strengthIndicator) {
                strengthIndicator = document.createElement('div');
                strengthIndicator.id = 'password-strength';
                strengthIndicator.className = 'form-text';
                passwordInput.parentNode.insertBefore(strengthIndicator, passwordInput.nextSibling);
            }

            strengthIndicator.textContent = `Password strength: ${strengthText}`;
            strengthIndicator.className = `form-text ${strengthClass}`;
        });
    }

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
});
