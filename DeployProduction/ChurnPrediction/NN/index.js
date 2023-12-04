async function runExample() {
  // Retrieve user inputs from HTML input fields
  var x = [];
  x[0] = parseFloat(document.getElementById('box1').value);
  x[1] = parseFloat(document.getElementById('box2').value);
  x[2] = parseFloat(document.getElementById('box3').value);
  x[3] = parseFloat(document.getElementById('box4').value);
  x[4] = parseFloat(document.getElementById('box5').value);
  x[5] = parseFloat(document.getElementById('box6').value);
  x[6] = parseFloat(document.getElementById('box7').value);
  x[7] = parseFloat(document.getElementById('box8').value);
  x[8] = parseFloat(document.getElementById('box9').value);
  x[9] = parseFloat(document.getElementById('box10').value);
  x[10] = parseFloat(document.getElementById('box11').value);
  x[11] = parseFloat(document.getElementById('box12').value);
  x[12] = parseFloat(document.getElementById('box13').value);
  x[13] = parseFloat(document.getElementById('box14').value);
  x[14] = parseFloat(document.getElementById('box15').value);
  x[15] = parseFloat(document.getElementById('box16').value);
  x[16] = parseFloat(document.getElementById('box17').value);

  // Create an ONNX Tensor with user inputs
  let tensorX = new onnx.Tensor(x, 'float32', [1, 17]);

  let session = new onnx.InferenceSession();

  await session.loadModel("churn_prediction.onnx");
  let outputMap = await session.run([tensorX]);
  let outputData = outputMap.get('output1');

  // Round the output value to the nearest integer
  let roundedOutput = Math.round(outputData.data[0]);
  console.log('Model exported to ONNX:', onnxPath);
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
