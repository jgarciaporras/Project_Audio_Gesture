var audio= document.getElementById('audio');
var playPauseBTN = document.getElementById('playPauseBTN');
var count=0;



function playPause(){
	if(count == 0){
		count = 1;
		audio.play();
        playPauseBTN.innerHTML = "Pause &#9208;"
    }else{
        count = 0;
		audio.pause();
        playPauseBTN.innerHTML = "Play &#9658;"
    }
}

function Stop(){
    playPause()
    audio.pause();
    audio.currentTime= 0;
    playPauseBTN.innerHTML = "Play &#9658;"

}

function Volume_20(){
    audio.volume = 0.2;
}
function Volume_40(){
    audio.volume = 0.4;
}
function Volume_60(){
    audio.volume = 0.6;
}

function Volume_80(){
    audio.volume = 0.8;
}

function Volume_max(){
    audio.volume = 1;
}
function Mute(){
    audio.volume = 0;
}


