var mousePressed = false;
var lastX, lastY;
var ctx;

function InitThis() {
    ctx = document.getElementById('myCanvas').getContext("2d");

    $('#myCanvas').mousedown(function (e) {
        mousePressed = true;
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas').mousemove(function (e) {
        if (mousePressed) {
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas').mouseleave(function (e) {
        mousePressed = false;
    });
}

function Draw(x, y, isDown) {
    if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = $('#selColor').val();
        ctx.lineWidth = $('#selWidth').val();
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x; lastY = y;
}

function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

function submit_pixels(canvas) {
    $('form input[name=image_data]').val(canvas.toDataURL("image/png"));
    $('form input[name=width]').val(canvas.width);
    $('form input[name=height]').val(canvas.height);
    $('form').submit();
};


// document.addEventListener("DOMContentLoaded", function() {
//     var butt1 = document.querySelector('#button1');
//
//     butt1.addEventListener('click', function (event) {
//             // document.getElementById('myCanvas').value = canvas.toDataURL('image/png');
//             // document.forms["form1"].submit();
//         $('form input[name=image_data]').val(canvas.toDataURL("image/png"));
//         $('form input[name=width]').val(canvas.width);
//         $('form input[name=height]').val(canvas.height);
//         $('form').submit();
//
//
//
//     })


// });