async function runExample() {
  // Retrieve user inputs from HTML input fields
  var x = [];
  for (let i = 1; i <= 17; i++) {
    x.push(parseFloat(document.getElementById('box' + i).value));
  }

  // Create an ONNX Tensor with user inputs
  let tensorX = new onnx.Tensor(x, 'float32', [1, 17]);

  let session = new onnx.InferenceSession();

  await session.loadModel("churn_prediction.onnx");
  let outputMap = await session.run([tensorX]);
  let outputData = outputMap.values().next().value;

  // Round the output value to the nearest integer
  let roundedOutput = (outputData.data[0]);

  // Display predictions on the web page
  let predictions = document.getElementById('predictions');
  predictions.innerHTML = `
    <hr> Churn Prediction Result: <br />
    <table>
      <tr>
        <td>Customer exited or not</td>
        <td id="result">${roundedOutput}</td>
      </tr>
    </table>
  `;
}
