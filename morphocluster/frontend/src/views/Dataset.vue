<template>
    <div id="dataset" class="view">
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
                <template v-if="dataset">
                    <!--{{dataset}}-->
                    <v-row>
                        <v-btn small :to="{name:'dataset-edit', dataset_id: dataset.dataset_id}">
                            <v-icon>mdi-cogs</v-icon>Settings
                        </v-btn>
                    </v-row>
                    <h2>Projects</h2>
                    <projects-table :dataset_id="dataset_id" />
                    <!--
                <h2>Expert Mode</h2>
                <div style="margin: 0 auto; width: 0;">
                    <b-button size="sm" variant="danger" class="mr-2" href="/labeling">Expert mode</b-button>
                </div>
                    -->
                </template>
            </div>
        </div>
    </div>
</template>

<script>
import * as api from "@/helpers/api.js";
import mixins from "@/mixins.js";

import ProjectsTable from "@/components/ProjectsTable.vue";

export default {
    name: "dataset",
    props: { dataset_id: Number },
    components: { ProjectsTable },
    mixins: [mixins],
    data() {
        return {
            dataset: null,
            alerts: []
        };
    },
    methods: {},
    created() {
        // Load node info
        api.getDataset(this.dataset_id)
            .then(data => {
                this.dataset = data;

                // Update breadcrumb
                this.setBreadcrumbs([
                    {
                        text: this.dataset.name,
                        to: {
                            name: "dataset",
                            props: { dataset_id: this.dataset.dataset_id }
                        }
                        //exact: true
                    }
                ]);
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style>
#dataset {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}
</style>
