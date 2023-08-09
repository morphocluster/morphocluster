<template>
    <div id="files">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark text-light">
            <ul class="navbar-nav mr-5">
                <li class="nav-item active text-light">Files</li>
            </ul>
            <router-link class="navbar-brand text-light mr-5" to="/">MorphoCluster</router-link>
            <router-link class="navbar-brand text-light mr-auto" to="/">Projects</router-link>
            <dark-mode-control />
        </nav>
        <div class="scrollable">
            <div class="container">
                <div class="alerts" v-if="alerts.length">
                    <b-alert :key="a" v-for="a of alerts" dismissible show :variant="a.variant">
                        {{ a.message }}
                    </b-alert>
                </div>
                <b-table id="files_table" striped sort-by="name" :items="projects" :fields="fields" showEmpty>

                </b-table>
            </div>
        </div>
    </div>
</template>

<script>

import * as api from "@/helpers/api.js";
import Humanize from "humanize-plus";
import DarkModeControl from "@/components/DarkModeControl.vue";



export default {
    name: "FilesView",
    props: {},
    components: { DarkModeControl },
    data() {
        return {
            fields: [
                { key: "name", sortable: true }
            ],
            files: [],
            alerts: [],
            Humanize,
        };
    },
    methods: {

    },
    mounted() {
        api.getFiles()
            .then((files) => {
                this.files = files
            }).catch((e) => {
                console.log(e);
                // TODO: Use axiosErrorHandler
                this.alerts.unshift({
                    message: e.message,
                    variant: "danger",
                });
            });
    }
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
</style>