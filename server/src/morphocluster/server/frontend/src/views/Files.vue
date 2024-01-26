<template>
    <div id="files">
        <div class="scrollable">
            <h1>{{ this.breadcrumb }}</h1>

            <div class="container" v-if="this.entry.type === 'directory'">
                <div class="alerts" v-if="alerts.length">
                    <v-alert :key="a" v-for="a of alerts" dismissible show :variant="a.variant">
                        {{ a.message }}
                    </v-alert>
                </div>

                <v-data-table :items="entry.children" :headers="headers" hide-default-footer>
                    <template v-slot:[`item.name`]="{ item }">
                        <router-link v-if="item.type === 'directory'"
                            :to="{ name: 'files', params: { file_path: item.path } }">
                            <i class="mdi mdi-folder" /> {{ item.name }}
                        </router-link>
                        <router-link v-if="item.type === 'file'" :to="{ name: 'files', params: { file_path: item.path } }">
                            <i class="mdi mdi-file" /> {{ item.name }}
                        </router-link>
                    </template>
                </v-data-table>

                <div class="dropzone" @dragover.prevent @dragenter.prevent @dragleave.prevent @drop="handleDrop">
                    Upload Files
                </div>
                <input type="file" id="fileInput" style="display: none" @change="handleFileSelect" multiple />
                <div class="container mt-4 text-center">
                    <v-btn large color="primary" class="mr-2" @click="openFileInput">Select File</v-btn>
                </div>
            </div>
            <div class="container" v-if="this.entry.type === 'file'">
                <v-card>
                    <v-card-title>Entry Information</v-card-title>
                    <v-card-text>
                        <v-row style="line-height: 1.5;">
                            <v-col cols=" 6">
                                <strong>Name:</strong>
                            </v-col>
                            <v-col cols="6">
                                <strong style="font-weight: bold;">{{ entry.name }}
                                </strong> </v-col>
                        </v-row>
                        <v-row>
                            <v-col cols="6">
                                <strong>Created On:</strong>
                            </v-col>
                            <v-col cols="6">
                                {{ entry.last_modified }}
                            </v-col>
                        </v-row>
                    </v-card-text>
                </v-card>
                <div class=" d-flex justify-content-center">
                    <v-row justify="center" class="my-2">
                        <v-btn color="primary" class="mr-2" @click.prevent="downloadFile">
                            Download
                        </v-btn>
                        <v-btn color="primary" class="mr-2">
                            Import as project
                        </v-btn>
                    </v-row>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import "@mdi/font/css/materialdesignicons.css";
import * as api from "@/helpers/api.js";
import mixins from "@/mixins.js";

export default {
    name: "FilesView",
    mixins: [mixins],
    props: { file_path: String },
    response: "",
    data() {
        return {
            headers: [
                { text: "Name", value: "name" },
                { text: "Last Modified", value: "last_modified" }
            ],
            entry: null,
            alerts: [],
        };
    },
    mounted() {
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
                this.breadcrumb = this.entry.path.split('/');
                if (this.breadcrumb.includes(".")) {
                    this.breadcrumb = []; // Oder eine andere Aktion, um das unerwünschte Element zu entfernen
                }
                this.breadcrumb = ["files"].concat(this.breadcrumb);
                this.setBreadcrumbs(this.breadcrumb, "files");
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
            const response = await api.uploadFiles(formData, this.entry.path);
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