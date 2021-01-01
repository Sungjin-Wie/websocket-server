const SocketServer = require("websocket").server;
const http = require("http");
let { PythonShell } = require("python-shell");
var express = require('express');
var app = express();

// const server = http.createServer((req, res) => {
//   console.log((new Date()) + " Received request for " + req.url);
//   res.writeHead(200);
//   res.end();
// });

var server = http.createServer(app).listen(80, function () {
  console.log('foo');
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
    console.log(connections);
    console.log(mes);
    PythonShell.run(
      "/home/ubuntu/server/Inference/Inference.py",
      null,
      (err, data) => {
        if (err) {
          console.log("error");
          console.log(err);
          console.log(data);
        } else {
          console.log("succeeded");
          console.log(data);
        }
      }
    );
    connections.forEach((element) => {
      if (element != connection) element.sendUTF(mes.utf8Data);
    });
  });

  connection.on("close", (resCode, des) => {
    console.log("connection closed");
    connections.splice(connections.indexOf(connection), 1);
  });
});
