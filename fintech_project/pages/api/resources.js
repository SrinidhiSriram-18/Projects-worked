const AWS = require("aws-sdk");

export default (req, res) => {
    if (req.method === "POST") {
        console.log(req).body;

        res.statusCode = 200;
        if (req.files.file.path) {
            // const tmp_path = req.body.file.path;
            const content_type = req.files.file.type;

            const documentType = req.body.type;

            console.log(tmp_path, content_type, documentType);
        }

        //   var directory = req.body.category || "default";

        //   const params = {
        //     Bucket: "sure-hired/" + userId + "/" + documentType,
        //     Key: `${image_name}`,
        //     ACL: "public-read",
        //     Body: image,
        //     ContentType: content_type
        //   };

        //   s3.upload(params, function(err, data) {
        //     if (err) return next(new api_error(err));

        //     if ("Location" in data) {
        //       res.send(data);
        //     } else {
        //       next(new api_error("Missing Resource URL"));
        //     }
        //   });
        // } else {
        //   next(new api_error("File Missing"));
        // }
    }
};