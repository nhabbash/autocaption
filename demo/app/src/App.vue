<template>
  <div id="app">
    <h1>Autocaption</h1>
    <h4>Automatic image caption generation</h4>
    <div class="container">
      <div class="row">
        <div class="col-sm picture">
          <picture-input 
            ref="pictureInput" 
            @change="onChange" 
            width="400" 
            height="400" 
            margin="16" 
            accept="image/jpeg,image/png" 
            size="10" 
            button-class="btn-sm btn-secondary" 
            removeButtonClass="btn-sm btn-secondary" 
            :removable="true"
            :customStrings="{
              upload: '<h1>Bummer!</h1>',
              drag: 'Choose a picture'
            }">
          </picture-input>
          <button class="btn btn-sm btn-light" v-on:click="submitFile()">Generate caption</button>
        </div>
      <div class="col-sm d-flex align-items-center justify-content-center">
        <Spinner v-if="this.loading==true" size="medium" />
        <ol>
          <li v-for="item in captions" :key="item">
            {{ item }}
          </li>
        </ol>
      </div>
    </div>
  </div>
    <a href="https://github.com/nhabbash/autocaption" class="text-decoration-none">View project on GitHub</a>
  </div>
</template>

<script>
import axios from 'axios';
import PictureInput from 'vue-picture-input'
import Spinner from 'vue-simple-spinner'

export default {
  name: 'app',
  data () {
    return {
      captions: [],
      image: '',
      loading: false
    }
  },
  components: {
    PictureInput,
    Spinner
  },

  methods: {
    onChange (image) {
      this.captions = []
      console.log('New picture selected!')
      if (image) {
        console.log('Picture loaded.')
        this.image = this.$refs.pictureInput.file;
      } else {
        console.log('FileReader API not supported: use the <form>, Luke!')
      }
    },
    submitFile() {
      if (this.image) {
        let formData = new FormData();
        formData.append('file', this.image);
        this.loading=true
        axios.post( '/predict',
          formData,
          {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
          }
        ).then((response) => {
          this.captions = response.data.candidates;
          this.loading=false;
        })
        .catch(error => {
          console.log(error);
          this.loading=false;
        });

      } else {
        console.log("No picture uploaded.")
      }
    }
  }
}
</script>

<style>
@import "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css";
@import url('https://fonts.googleapis.com/css2?family=Lato&display=swap');

html, body {
  font-family: 'Lato', sans-serif;
}

html {
  background-color: #222A33;
  height: 100%;
}
#app {
  background-color: #222A33;
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #D3D3D3;
  margin-top: 60px;
}

h1, h2 {
  font-weight: normal;
}

picture ul {
  list-style-type: none;
  padding: 0;
}

picture li {
  display: inline-block;
  margin: 0 10px;
}

a {
  color: #42b983;
}
</style>

