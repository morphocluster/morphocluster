<template>
    <div id="files">
        <nav class="navbar navbar-expand-lg navbar navbar-dark bg-dark">
            <a class="navbar-brand" href="/p">MorphoCluster</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav"
                aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation" >
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/p">Projects</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" >Files/{{ this.file_path }}<span class="sr-only">(current)</span></a>
                    </li>
                </ul> 
            </div>
        </nav>
        <div class="scrollable">
            <div class="container">
                <div class="alerts" v-if="alerts.length">
                    <b-alert :key="a" v-for="a of alerts" dismissible show :variant="a.variant">
                        {{ a.message }}
                    </b-alert>
                </div>
                <b-table id="files_table" striped sort-by="name" :items="files" :fields="fields" showEmpty>                
                    <template v-slot:cell(name)="data">
                        <router-link v-if="data.item.Type === 'directory' " :to="{
                            name: 'files',
                            params: { file_path:  data.item.Path},
                        }">{{ data.item.Name }}</router-link>
                        <div v-if="data.item.Type === 'file'" class="file-link">
                            {{data.item.Name}}
                        </div>
                    </template>
                </b-table>
            </div>
        </div>
    </div>
</template>

<script>



import axios from "axios";

import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap";


export default {
    name: "FilesView",
    props: { "file_path": String},
    components: { },
    test: "test",

    data() {
        return {
            fields: [
                { key: "Name", sortable: true },
                "Last_modified",
            ],
            files: [],
            alerts: [],
            
        };
    },
    created() {
        this.initialize();
    },
    watch: {
        $route: "initialize",
    },
    methods: {
        async initialize() {
            try{
                const response = await axios.get(`/api/files/${this.file_path}`);
                this.files = response.data;
                this.test = this.files[0]["Name"];
            }catch(error){
                console.error(error);
                this.alerts.unshift({
                    message: error.message,
                    variant: "danger",
                });
            }
        }
    },
}




</script>

<style>
#files {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}

#files_table tr td:nth-child(1) {
    width: 100%;
}

#files_table tr td:not(:nth-child(1)) {
    width: auto;
    text-align: right;
    white-space: nowrap;
}

.scrollable {
    overflow-y: auto;
}

.alerts {
    padding-top: 1em;
}
.file-link {
  color: rgb(9, 107, 41); /* Ändere die Farbe nach Bedarf */
}
</style>