const SocketServer = require("websocket").server;
const http = require("http");
let { PythonShell } = require("python-shell");

const server = http.createServer((req, res) => {});

server.listen(3000, () => {
  console.log("Listening on port 3000...");
});

wsServer = new SocketServer({ httpServer: server });

const connections = [];

wsServer.on("request", (req) => {
  const connection = req.accept();
  console.log("new connection");
  connections.push(connection);
  console.log(connections);

  connection.on("message", (mes) => {
    PythonShell.run(
      "/Users/sungjin/dev/socketserver/Inference/Inference.py",
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
