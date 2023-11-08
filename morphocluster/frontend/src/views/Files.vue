<template>
    <div id="files">
        <nav class="navbar navbar-expand-lg navbar navbar-dark bg-dark">
            <a class="navbar-brand" href="/p">MorphoCluster</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav"
                aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/p">Projects</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active">Files/{{ this.file_path }}<span class="sr-only">(current)</span></a>
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
                        <router-link v-if="data.item.type === 'directory'" :to="{
                            name: 'files',
                            params: { file_path: data.item.path },
                        }"><i class="mdi mdi-folder" />  {{ data.item.name }}</router-link>
                        <router-link v-if="data.item.type === 'file'" :to="{
                            name: 'file',
                            params: { file_name: data.item.name, file_path: data.item.path },
                        }"><i class="mdi mdi-file" />  {{ data.item.name}} </router-link>
                    </template>
                </b-table>
            </div>
        </div>
        <div class="container mt-4">
            <div class="dropzone" @dragover.prevent @dragenter.prevent @dragleave.prevent @drop="handleDrop">
                Upload Files
            </div>
            <input type="file" id="fileInput" style="display: none" @change="handleFileSelect" multiple />
            <div class="container mt-4 text-center">
                <button class="btn btn-primary" @click="openFileInput">Select File</button>
            </div>

        </div>
    </div>

</template>

<script>
import "@mdi/font/css/materialdesignicons.css";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap";
import { uploadFiles } from "../helpers/api.js";

export default {
    name: "FilesView",
    props: { file_path: String },
    components: {},
    response: "",
    data() {
        return {
            fields: [
                { key: "name", sortable: true },
                "last_modified",
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
        openFileInput() {
            document.getElementById("fileInput").click();
        },
        async initialize() {
            try {
                const response1 = await axios.get(`/api/files/${this.file_path}?download=false&info=true`);
                this.files = response1.data["children"];
            } catch (error) {
                console.error(error);
                this.alerts.unshift({
                    message: error.message,
                    variant: "danger",
                });
            }
        },
        async handleDrop(event) {
            this.uploadFiles(event, true);
        },
        async handleFileSelect(event) {
            this.uploadFiles(event, false);
        },
        async uploadFiles(event,is_drop) {
            event.preventDefault();
            const selectedFiles = is_drop ? event.dataTransfer.files : event.target.files;
            const formData = new FormData();
            for(let i = 0; i < selectedFiles.length; i++){
                const file = selectedFiles[i];
                formData.append('file',file);
            }
            const response = await uploadFiles(formData, this.file_path);
            console.log("Data upload successful", response.message);
            this.initialize();
        },
    },
};
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

.dropzone {
    border: 2px dashed #ccc;
    padding: 20px;
    text-align: center;
    cursor: pointer;
}
</style>