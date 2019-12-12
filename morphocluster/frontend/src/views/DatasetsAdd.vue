<template>
    <div id="datasets">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark">
            <router-link class="navbar-brand text-light" to="/"
                >MorphoCluster</router-link
            >
            <ul class="navbar-nav mr-auto">
                <li class="nav-item active text-light">Add dataset</li>
            </ul>
        </nav>
        <div class="scrollable">
            <div class="container">
                <div class="alerts" v-if="alerts.length">
                    <b-alert
                        :key="a"
                        v-for="a of alerts"
                        dismissible
                        show
                        :variant="a.variant"
                        >{{ a.message }}</b-alert
                    >
                </div>
                <file-upload ref="upload" v-model="files"
                    >Upload file</file-upload
                >
            </div>
        </div>
    </div>
</template>

<script>
import "@mdi/font/css/materialdesignicons.css";
//import * as api from "@/helpers/api.js";

import VueUploadComponent from "vue-upload-component";

export default {
    name: "datasets",
    props: {},
    components: { FileUpload: VueUploadComponent },
    data() {
        return {
            fields: [
                { key: "name", sortable: true }
                //"action"
            ],
            alerts: []
        };
    },
    methods: {},
    mounted() {}
};
</script>

<style>
#datasets {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}

#datasets_table tr td:nth-child(1) {
    width: 100%;
}

#datasets_table tr td:not(:nth-child(1)) {
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
</style>
