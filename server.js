const SocketServer = require("websocket").server;
const http = require("http");
let { PythonShell } = require("python-shell");
var express = require('express');
var app = express();
let fs = require('fs');
let wavConverter = require('wav-converter');

var data = fs.readFileSync("./Inference/cafeafter.wav");
console.log(data);

// var pcmData = wavConverter.decodeWav(data);
// console.log(pcmData);

// var wavData = wavConverter.encodeWav(pcmData, {
//   numChannels: 1,
//   sampleRate: 16000,
//   byteRate: 16
// });
// console.log(wavData);

// const server = http.createServer((req, res) => {
//   console.log((new Date()) + " Received request for " + req.url);
//   res.writeHead(200);
//   res.end();
// });

var port = 3000;

function base64encode(plaintext){
  return Buffer.from(plaintext, "utf8").toString('base64');
}

function base64decode(base64text){
  return Buffer.from(base64text, 'base64');
}


var server = http.createServer(app).listen(port, function () {
  console.log(`listening on port ${port}`);
});

app.get("/", (req, res) => {
  res.send("foo");
});

wsServer = new SocketServer({ httpServer: server });

const connections = [];

wsServer.on("request", (req) => {
  const connection = req.accept();
  console.log("new connection");
  connections.push(connection);
  console.log(connections);

  connection.on("message", (mes) => {
    console.log(mes);
    console.log("received base64 string, decoding..")
    let buf = base64decode(mes.utf8Data);
    console.log("decoded base64 to buffer")
    console.log(buf);
    var wavData = wavConverter.encodeWav(buf, {
      numChannels: 1,
      sampleRate: 16000,
      byteRate: 16
    });
    fs.writeFileSync('./before.wav', wavData);
    PythonShell.run(
      "/home/ubuntu/server/Inference/Inference.py",
      null,
      (err, data) => {
        after = data;
        if (err) {
          console.log("error");
          console.log(err);
          console.log(data);
        } else {
          console.log("succeeded");
          console.log(data);
          let after = fs.readFileSync("./after.wav");
          console.log("readed afterData");
          console.log(after);
          let decodedAfterToPCM = wavConverter.decodeWav(after);
          console.log("decoded to pcm");
          console.log(decodedAfterToPCM);
          let encodedAfterToBase64 = base64encode(decodedAfterToPCM);
          console.log("encoded to base64 text");
          console.log(encodedAfterToBase64);
          console.log(connections);
          connections.forEach((element) => {
            // if (element != connection) 
            element.sendUTF(encodedAfterToBase64);
          });
        }
      }
    );
  });

  connection.on("close", (resCode, des) => {
    console.log("connection closed");
    connections.splice(connections.indexOf(connection), 1);
  });
});
