document.getElementById('flight-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var airline = document.getElementById('airline').value;
    // Other form data retrieval...

    var data = {
        // Form data...
    };

    // Log the data before sending
    console.log('Data being sent:', JSON.stringify(data));

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('result').innerHTML = `
            <p>Delay Probability: ${data.delay_probability}</p>
            <p>Delay: ${data.delay}</p>
        `;
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
            // Handle/display the error to the user
            document.getElementById('result').innerHTML = '<p>Error fetching data. Please try again later.</p>';
        });
});
