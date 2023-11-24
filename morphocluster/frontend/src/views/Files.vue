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
                        <router-link class="nav-link" :to="{ name: 'projects' }">Projects</router-link>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active">Files<span class="sr-only">(current)</span></a>
                    </li>
                </ul>
            </div>
        </nav>
        <div class="container">
            <nav aria-label="breadcrumb">
                <ol class="breadcrumb">
                    <li class="breadcrumb-item">
                        <router-link :to="{ name: 'files' }">Home</router-link>
                    </li>
                    <li v-for="(path, index) in breadcrumb" :key="index" class="breadcrumb-item">
                        <router-link :to="{ name: 'files', params: { file_path: path } }">{{ getBasename(path)
                        }}</router-link>
                    </li>
                </ol>
            </nav>
        </div>
        <div class="scrollable">
            <div class="container" v-if="this.file.type === 'directory'">
                <div class="alerts" v-if="alerts.length">
                    <b-alert :key="a" v-for="a of alerts" dismissible show :variant="a.variant">
                        {{ a.message }}
                    </b-alert>
                </div>
                <b-table id="files_table" striped :items="files" :fields="fields" showEmpty>
                    <template v-slot:cell(name)="data">
                        <router-link v-if="data.item.type === 'directory'" :to="{
                            name: 'files',
                            params: { file_path: data.item.path },
                        }"><i class="mdi mdi-folder" /> {{ data.item.name }}</router-link>
                        <router-link v-if="data.item.type === 'file'" :to="{
                            name: 'files',
                            params: { file_path: data.item.path },
                        }"><i class="mdi mdi-file" /> {{ data.item.name }} </router-link>
                    </template>
                </b-table>
                <div class="dropzone" @dragover.prevent @dragenter.prevent @dragleave.prevent @drop="handleDrop">
                    Upload Files
                </div>
                <input type="file" id="fileInput" style="display: none" @change="handleFileSelect" multiple />
                <div class="container mt-4 text-center">
                    <button class="btn btn-primary" @click="openFileInput">Select File</button>
                </div>
            </div>
            <div class="container" v-if="this.file.type === 'file'">
                <b-table v-if="this.file.type === 'file'" id="files_table" stacked :items="fileInfos"></b-table>
                <div class="d-flex justify-content-center">
                    <b-button size="sm" variant="primary" class="mx-2" @click.prevent="downloadFile">
                        Download
                    </b-button>
                    <b-button size="sm" variant="primary" class="mx-2">
                        Import as project
                    </b-button>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import "@mdi/font/css/materialdesignicons.css";
import * as api from "@/helpers/api.js";
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
                { key: "name" },
                "last_modified",
            ],
            files: [],
            file: {},
            fileInfos: [],
            breadcrumb: null,
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
        getBasename(path) {
            const parts = path.split('/');
            return parts[parts.length - 1];
        },
        openFileInput() {
            document.getElementById("fileInput").click();
        },
        async initialize() {
            try {
                const data = await api.getFileInfo(this.file_path);
                const { children, breadcrumb_paths, ...fileinfo } = data;
                this.breadcrumb = breadcrumb_paths;
                this.files = children;
                this.file = fileinfo;
                this.fileInfos = [fileinfo];
            } catch (error) {
                console.error(error);
                this.alerts.unshift({
                    message: error.message,
                    variant: "danger",
                });
            }
        },
        async handleDrop(event) {
            event.preventDefault();
            this.uploadFiles(event.dataTransfer.files);
        },
        async handleFileSelect(event) {
            event.preventDefault();
            this.uploadFiles(event.target.files);
        },
        async uploadFiles(selectedFiles) {
            const formData = new FormData();
            for (let i = 0; i < selectedFiles.length; i++) {
                const file = selectedFiles[i];
                formData.append('file', file);
            }
            const response = await uploadFiles(formData, this.file_path);
            console.log("Data upload successful", response.message);
            this.initialize();
        },
        downloadFile() {
            if (this.fileInfos.length > 0) {
                window.open(`/api/files/${this.file.path}?download=1`);
            } else {
                this.alerts.unshift({
                    message: "No file available for download.",
                    variant: "danger",
                });
            }
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