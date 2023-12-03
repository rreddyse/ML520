async function runExample() {
  var x = new Float32Array(1, 14);

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

  let tensorX = new onnx.Tensor(x, 'float32', [1, 14]);

  let session = new onnx.InferenceSession();

  await session.loadModel("Net_ChurnData.onnx");
  let outputMap = await session.run([tensorX]);
  let outputData = outputMap.get('output1');

  let predictions = document.getElementById('predictions');
  predictions.innerHTML = ` <hr> Got an output tensor with value: <br />
  <table>
      <tr>
          <td>Customer exited or not</td>
          <td id="td0"> ${outputData.data[0].toFixed(2)} </td>

      </tr>
  </table
  `;
}
