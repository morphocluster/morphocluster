<template>
    <div id="files">
        <nav class="navbar navbar-expand-lg navbar navbar-dark bg-dark">
            <router-link class="navbar-brand" :to="{ name: 'home' }">MorphoCluster</router-link>
            <div class="navbar-collapse" id="navbarNav">
                <ul class="navbar-nav nav-item">
                    <li class="navbar-item">
                        <router-link class="nav-link" :to="{ name: 'files' }">Files</router-link>
                    </li>
                </ul>
                <ul class="navbar-nav nav-item">
                    <li v-for="(parent, index) in entry.parents.slice()" :key="index" class="navbar-item">
                        <router-link class="nav-link" :to="{ name: 'files', params: { file_path: parent.path } }">{{
                            parent.name
                        }}</router-link>
                    </li>
                    <li class="navbar-item" v-if="this.entry.name != '.'">
                        <span class="nav-link">{{ this.entry.name }}</span>
                    </li>
                </ul>
            </div>
            <dark-mode-control />
        </nav>
        <div class="scrollable">
            <div class="container" v-if="this.entry.type === 'directory'">
                <div class="alerts" v-if="alerts.length">
                    <b-alert :key="a" v-for="a of alerts" dismissible show :variant="a.variant">
                        {{ a.message }}
                    </b-alert>
                </div>
                <b-table id="files_table" striped :items="entry.children" :fields="fields" showEmpty>
                    <template v-slot:cell(name)="child">
                        <router-link v-if="child.item.type === 'directory'" :to="{
                            name: 'files',
                            params: { file_path: child.item.path },
                        }"><i class="mdi mdi-folder" /> {{ child.item.name }}</router-link>
                        <router-link v-if="child.item.type === 'file'" :to="{
                            name: 'files',
                            params: { file_path: child.item.path },
                        }"><i class="mdi mdi-file" /> {{ child.item.name }} </router-link>
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
            <div class="container" v-if="this.entry.type === 'file'">
                <!--TODO: Convert to regular table and select and format properties by hand. -->
                <table id="table" style="width=100%">
                    <tr>
                        <td style="padding-right: 20px;">Name:</td>
                        <td>{{ this.entry.name }}</td>
                    </tr>
                    <tr>
                        <td style="padding-right: 20px;">Created On:</td>
                        <td>{{ this.entry.last_modified }}</td>
                    </tr>
                </table>
                <div class=" d-flex justify-content-center">
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
import DarkModeControl from "@/components/DarkModeControl.vue";

export default {
    name: "FilesView",
    props: { file_path: String },
    response: "",
    components: { DarkModeControl },
    data() {
        return {
            fields: [
                { key: "name" },
                "last_modified",
            ],
            entry: null,
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
                this.entry = await api.getFileInfo(this.file_path);
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
            const response = await uploadFiles(formData, this.entry.path);
            console.log("Data upload successful", response.message);
            this.initialize();
        },
        downloadFile() {
            window.open(`/api/files/${this.entry.path}?download=1`);
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