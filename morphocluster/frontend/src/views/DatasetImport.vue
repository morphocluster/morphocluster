<template>
    <div id="datasets">
        <div class="scrollable">
            <div class="container">
                <div class="alerts" v-if="alerts.length">
                    <b-alert
                        :key="a"
                        v-for="a of alerts"
                        dismissible
                        show
                        :variant="a.variant"
                    >{{ a.message }}</b-alert>
                </div>
                <file-upload ref="upload" v-model="files">Upload file</file-upload>
            </div>
        </div>
    </div>
</template>

<script>
//import * as api from "@/helpers/api.js";
import mixins from "@/mixins.js";

import VueUploadComponent from "vue-upload-component";

export default {
    name: "dataset-create",
    props: {},
    components: { FileUpload: VueUploadComponent },
    mixins: [mixins],
    data() {
        return {
            fields: [
                { key: "name", sortable: true }
                //"action"
            ],
            files: [],
            alerts: []
        };
    },
    methods: {},
    mounted() {
        this.setBreadcrumbs([
            {
                text: "Create dataset"
            }
        ]);
    }
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
