const escapeHtml = require('escape-html')

const {
  SIGN_KIND,
  SIGNS_PROCESSED_BUCKET_NAME,
  buildAuthorizationHeader
} = require('./common')
const datastore = require('./datastore')
const storage = require('./storage')

const LIMIT = 30

const SIGNS_APPROVE_URL = `https://europe-west1-my-eu-1532800860795.cloudfunctions.net/signs_approve`
const SIGNS_PROCESS_URL = `https://europe-west1-my-eu-1532800860795.cloudfunctions.net/signs_process`

exports.admin = function signsAdmin(req, res) {
  const query = datastore
    .createQuery(SIGN_KIND)
    .filter('approved', '=', null)
    .limit(LIMIT)
  return datastore
    .runQuery(query)
    .then(getSignedUrls)
    .then(signs => {
      res.send(renderSigns(signs))
    })
    .catch(error => {
      console.error(error)
      res.status(500).send({ message: 'failed to list' })
    })
}

function getSignedUrls(results) {
  const bucket = storage.bucket(SIGNS_PROCESSED_BUCKET_NAME)
  return Promise.all(
    results[0].map(sign => {
      const fileName = sign[datastore.KEY].name
      return bucket
        .file(fileName)
        .getSignedUrl({
          action: 'read',
          expires: new Date().getTime() + 60 * 60 * 1000
        })
        .then(response => {
          sign.url = response[0]
          return sign
        })
    })
  )
}

function renderSigns(signs) {
  const password = process.env.MY_EU_SIGNS_ADMIN_PASSWORD
  const authorization = buildAuthorizationHeader('admin', password)
  return `
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
    <title>myeu.uk - signs - admin</title>
    <script>
      function myEuModerate(name, approve) {
        var $card = $('#my-eu-card-' + name)

        function toggleButtons(disabled) {
          $card.find('button').prop('disabled', disabled)
        }
        toggleButtons(true)

        var useExif = $card.find('input[type="radio"]:checked').val() === 'exif'

        fetch('${SIGNS_APPROVE_URL}', {
          mode: 'cors',
          method: 'POST',
          headers: {
            'Accept': 'application/json',
            'Authorization': '${authorization}',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            file_name: name, use_exif: useExif, approve: approve
          })
        }).then((response) => {
          if (response.status === 200) {
            $card.remove()
            return
          }
          throw new Error('Unexpected response: ' + response.status)
        }).catch((error) => {
          alert(error)
          toggleButtons(false)
        })
      }

      function myEuProcess(name) {
        var $card = $('#my-eu-card-' + name)

        function toggleButtons(disabled) {
          $card.find('button').prop('disabled', disabled)
        }
        toggleButtons(true)

        fetch('${SIGNS_PROCESS_URL}', {
          mode: 'cors',
          method: 'POST',
          headers: {
            'Accept': 'application/json',
            'Authorization': '${authorization}',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ file_name: name })
        }).then((response) => {
          if (response.status === 200) {
            window.location.reload()
            return
          }
          throw new Error('Unexpected response: ' + response.status)
        }).catch((error) => {
          alert(error)
          toggleButtons(false)
        })
      }
    </script>
  </head>
  <body>
    <div class="container">
      <h1>Unmoderated Signs</h1>
      ${signs.map(renderSign).join('\n')}
    </div>
    <script src="https://code.jquery.com/jquery-3.3.1.min.js" integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8=" crossorigin="anonymous"></script>
  </body>
</html>`
}

function renderSign(sign) {
  const fileName = sign[datastore.KEY].name
  const deviceLatlng = `${sign.deviceLatitude},${sign.deviceLongitude}`
  const exifLatlng = `${sign.exifLatitude},${sign.exifLongitude}`
  const exifDisabled = 'exifLatitude' in sign ? '' : ' disabled'
  return `
<div id="my-eu-card-${fileName}" class="card">
  <img src="${sign.url}" class="img-fluid">
  <div class="card-body text-center">
    <p>
      Title: &ldquo;${escapeHtml(sign.title)}&rdquo;
    </p>
    <p>
      <input type="radio" id="my-eu-location-device-${fileName}" name="location-${fileName}" value="device" checked>
      <label for="my-eu-location-device-${fileName}">
        <a href="https://www.google.com/maps/search/?api=1&query=${deviceLatlng}" target="_blank">
          Device Latitude, Longitude: ${escapeHtml(deviceLatlng)}
        </a>
      </label>
    </p>
    <p>
      <input type="radio" id="my-eu-location-exif-${fileName}" name="location-${fileName}" value="exif"${exifDisabled}>
      <label for="my-eu-location-exif-${fileName}">
        <a href="https://www.google.com/maps/search/?api=1&query=${exifLatlng}" target="_blank">
          EXIF Latitude, Longitude: ${escapeHtml(exifLatlng)}
        </a>
      </label>
    </p>
  </div>
  <div class="card-footer">
    <button class="btn btn-warning" onClick="myEuModerate('${fileName}', false)">Reject</button>
    <button class="btn btn-secondary" onClick="myEuProcess('${fileName}')">Reprocess</button>
    <button class="btn btn-success float-right" onClick="myEuModerate('${fileName}', true)">Approve</button>
  </div>
</div>`
}
