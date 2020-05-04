<template>
    <div id="datasets" class="view">
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
            <h1>Datasets</h1>
            <v-data-table
                id="datasets_table"
                sort-by="name"
                :items="datasets"
                :headers="headers"
                disable-filtering
                disable-pagination
                hide-default-footer
            >
                <template v-slot:no-data>No datasets.</template>
                <template v-slot:item.name="{ item }">
                    <router-link
                        :to="{
                                name: 'dataset',
                                params: { dataset_id: item.dataset_id }
                            }"
                    >{{ item.name }}</router-link>
                </template>
                <template slot="empty">
                    <div class="text-center">No datasets available.</div>
                </template>
            </v-data-table>
            <v-row>
                <v-spacer />
                <v-btn :to="{name: 'datasets-create'}">
                    <v-icon>mdi-plus</v-icon>Create dataset
                </v-btn>
            </v-row>
        </div>
    </div>
</template>

<script>
import * as api from "@/helpers/api.js";
import mixins from "@/mixins.js";

export default {
    name: "datasets",
    props: {},
    components: {},
    mixins: [mixins],
    data() {
        return {
            headers: [
                { value: "name", text: "Name", sortable: true }
                //"action"
            ],
            datasets: [],
            alerts: []
        };
    },
    methods: {},
    created() {
        this.setLoading("datasets");
        api.datasetsGetAll()
            .then(datasets => {
                this.datasets = datasets;
                // Update breadcrumb
                this.setBreadcrumbs([{ text: "Datasets" }]);
            })
            .catch(e => {
                console.log(e);
                this.alerts.unshift({
                    message: e.message,
                    variant: "danger"
                });
            })
            .finally(() => {
                this.unsetLoading("datasets");
            });
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
