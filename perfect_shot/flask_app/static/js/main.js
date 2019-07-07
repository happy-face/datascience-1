$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    var imgCount = 0

    // Upload Preview
    function readURL(input) {
        $('#imagePreviewContainer').empty();
        imgCount = 0;
        for (var i = 0; i < input.files.length; i++) {
            var reader = new FileReader();
            reader.onload = function (e) {
                id = 'imagePreview' + imgCount.toString();
                imgCount = imgCount + 1;
                $('#imagePreviewContainer').append('<div id="' + id + '"></div>');
                $('#' + id).css('background-image', 'url(' + e.target.result + ')');
                $('#' + id).hide();
                $('#' + id).fadeIn(650);

                // adjust aspect ratio
                var image = new Image();
                image.src = e.target.result;

                max_width = 256;
                max_height = 256;

                width = max_width;
                height = max_height;
                if (image.width > image.height)
                {
                    height = width * (image.height / image.width)
                }
                else
                {
                    width = height * (image.width / image.height)
                }

                $('#' + id).css('width', width.toString() + 'px');
                $('#' + id).css('height', height.toString() + 'px');
            }
            reader.readAsDataURL(input.files[i]);
            //Do something
        }
        // if (input.files && input.files[0]) {
        //     var reader = new FileReader();
        //     reader.onload = function (e) {
        //         $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
        //         $('#imagePreview').hide();
        //         $('#imagePreview').fadeIn(650);
        //     }
        //     reader.readAsDataURL(input.files[0]);
        // }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#resultContainer').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        console.log("btn-predict log");
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();
        $.ajax({
         type: 'POST',
         url: '/predict',
         data: form_data,
         contentType: false,
         cache: false,
         processData: false,
         async: true,
         success: function (data) {
           console.log(data.payload.toString());
           var sorted_list = JSON.parse(data.payload)
           console.log(sorted_list)

           var winner_id = 'imagePreview' + sorted_list[0].toString()
           $('#' + winner_id).clone().appendTo('#resultContainer');

           // Get and display the result
           $('.loader').hide();
           $('#result').fadeIn(600);
           //
           // var new_data = JSON.parse(data.payload);
           // $('#result').append('Prediction:');
           // for (var i in new_data){
           //   var _html = `
           //     <p>${new_data[i].name}</p>
           //
           //     <p>${new_data[i].val}</p>
           //   `
           //    $('#result').append(_html);
           // }
           console.log('Success!');
          },
         error: function(jqXHR, errorDesc, exp) {
           console.log(errorDesc);
           console.log(exp);
          },
        });
    });

});
