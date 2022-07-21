<template>
    <div id="datasets">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark">
            <router-link class="navbar-brand text-light" to="/"
                >MorphoCluster</router-link
            >
            <ul class="navbar-nav mr-auto">
                <li class="nav-item active text-light">Datasets</li>
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
                    >
                        {{ a.message }}
                    </b-alert>
                </div>
                <b-table
                    id="datasets_table"
                    striped
                    sort-by="name"
                    :items="datasets"
                    :fields="fields"
                    showEmpty
                >
                    <template slot="table-colgroup">
                        <col class="col-wide" />
                        <col class="col-narrow" />
                    </template>
                    <template v-slot:cell(name)="data">
                        <router-link
                            :to="{
                                name: 'projects',
                                params: { dataset_id: data.item.dataset_id },
                            }"
                            >{{ data.item.name }}</router-link
                        >
                    </template>
                    <!-- <template v-slot:cell(stats)="data">
                        <b-progress
                            v-if="'progress' in data.item"
                            :max="data.item.progress.leaves_n_nodes"
                        >
                            <b-progress-bar
                                variant="success"
                                :value="
                                    data.item.progress.leaves_n_filled_nodes
                                "
                                v-b-tooltip.hover
                                :title="`${data.item.progress.leaves_n_filled_nodes} / ${data.item.progress.leaves_n_nodes} filled up`"
                            />
                            <b-progress-bar
                                variant="warning"
                                :value="
                                    data.item.progress.leaves_n_approved_nodes -
                                    data.item.progress.leaves_n_filled_nodes
                                "
                                v-b-tooltip.hover
                                :title="`${data.item.progress.leaves_n_approved_nodes} / ${data.item.progress.leaves_n_nodes} approved`"
                            />
                        </b-progress>
                    </template> -->
                    <!-- <template v-slot:cell(action)="data">
                        <b-button
                            size="sm"
                            variant="primary"
                            class="mr-2"
                            :to="{
                                name: 'approve',
                                params: { dataset_id: data.item.dataset_id },
                            }"
                        >
                            Validate
                        </b-button>
                        <b-button
                            size="sm"
                            variant="primary"
                            class="mr-2"
                            :to="{
                                name: 'bisect',
                                params: { dataset_id: data.item.dataset_id },
                            }"
                            >Grow</b-button
                        >
                        <b-button
                            size="sm"
                            variant="primary"
                            class="mr-2"
                            @click.prevent="showSaveModal(data.item)"
                            >Save</b-button
                        >
                    </template> -->
                    <template v-slot:empty>
                        <div class="text-center">No datasets available.</div>
                    </template>
                </b-table>
            </div>
        </div>
    </div>
</template>

<script>
import * as api from "@/helpers/api.js";

export default {
    name: "datasets",
    props: { dataset_id: Number },
    components: {},
    data() {
        return {
            fields: [
                //{ key: "dataset_id", sortable: true },
                { key: "name", sortable: true },
                //"stats",
                //"action",
            ],
            datasets: [],
            alerts: [],
        };
    },
    mounted() {
        // Load node info
        api.getDatasets()
            .then((datasets) => {
                this.datasets = datasets;
            })
            .catch((e) => {
                console.log(e);
                this.alerts.unshift({
                    message: e.message,
                    variant: "danger",
                });
            });
    },
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
