async function runExample() {
 

  var x = [];
  x[0] =document.getElementById('box1').value;
  x[1] =document.getElementById('box2').value;
  x[2] =document.getElementById('box3').value;
  x[3] =document.getElementById('box4').value;
  x[4] =document.getElementById('box5').value;
  x[5] =document.getElementById('box6').value;
  x[6] =document.getElementById('box7').value;
  x[7] =document.getElementById('box8').value;
  x[8] =document.getElementById('box9').value;
  x[9] =document.getElementById('box10').value;
  x[10] =document.getElementById('box11').value;
  x[11] =document.getElementById('box12').value;
  x[12] =document.getElementById('box13').value;
  x[13] =document.getElementById('box14').value;
  x[14] =document.getElementById('box15').value;
  x[15] =document.getElementById('box16').value;
  x[16] =document.getElementById('box17').value;


  let tensorX = new onnx.Tensor(x, 'float32', [1, 17]);

let session = new onnx.InferenceSession();

await session.loadModel("Net_ChurnData1.onnx");
let outputMap = await session.run([tensorX]);
let outputData = outputMap.get('output1');

// Round the output value to the nearest integer
//let roundedOutput = Math.round(outputData.data[0]);

let predictions = document.getElementById('predictions');
predictions.innerHTML = ` <hr> Got an output tensor with value: <br />
  <table>
      <tr>
          <td>Customer exited or not</td>
          <td id="td0"> ${outputData.data[0].toFixed(2)} </td>
      </tr>
  </table>
  `;
}
