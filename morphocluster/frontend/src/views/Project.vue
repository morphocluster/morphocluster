<template>
    <div id="project">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark text-light">
            <ul class="navbar-nav mr-5">
                <li class="nav-item active text-light">Project</li>
            </ul>
            <router-link class="navbar-brand text-light mr-5" to="/"
                >MorphoCluster</router-link
            >
            <router-link class="navbar-brand text-light mr-5" to="/"
                >Projects</router-link
            >
            <router-link class="navbar-brand text-light mr-auto" :to="{name: 'files'}"
                >Files</router-link
            >
            <dark-mode-control />
        </nav>
        <div class="container">
            <b-table striped hover :items="items" :fields="fields"></b-table>
            <div style="margin: auto ; width: 0; padding-top: 7px ">
                <b-button size="sm" variant="primary" href="" @click.prevent="showSaveModal(project)">
                    Save Project
                </b-button>
            </div>
        </div>
        <b-modal ref="saveModal" lazy centered :title="save_title" @ok="HandleSaveOk">
            <div class="d-block text-center">
                <form @submit.stop.prevent="HandleSaveOk">
                    <b-container fluid>
                        <b-form-row class="my-1">
                            <b-col sm="3">
                                <label for="form_save_name">Name:</label>
                            </b-col>
                            <b-col sm="9">
                                <b-form-input type="text" placeholder="Enter a name for the saved file" id="form_save_name"
                                    v-model="save_slug"></b-form-input>
                            </b-col>
                        </b-form-row>
                        <b-row class="my-3" v-if="save_saving">
                            <b-col sm="12"> Saving... </b-col>
                        </b-row>
                    </b-container>
                </form>
            </div>
        </b-modal>
    </div>
</template>

<script>
import * as api from "@/helpers/api.js";
import axios from "axios";
import { EventBus } from "@/event-bus.js";
import DarkModeControl from "@/components/DarkModeControl.vue";

export default {
    name: "ProjectView",
    props: { "project_id": Number },
    components: { DarkModeControl },
    data() {
        return {
            project: null,
            save_slug: "",
            save_title: "",
            save_project_id: null,
            save_saving: false,
            fields: ['Category','Info'],
            items: []
        };
    },
    methods: {
        showSaveModal(project) {
            console.log("project", project);
            this.save_slug = project.name;
            this.save_project_id = project.project_id;
            this.save_title = `Save ${project.name} (${this.save_project_id})`;
            this.$refs.saveModal.show();
        },
        HandleSaveOk(evt) {
            evt.preventDefault();

            console.log("Saving", this.save_project_id, "...");
            this.save_saving = true;
            api.saveProject(this.save_project_id).then((result) => {
                console.log("Project saved: " + result["url"]);
                this.save_saving = false;
                this.$nextTick(() => {
                    // Wrapped in $nextTick to ensure DOM is rendered before closing
                    this.$refs.saveModal.hide();
                });
                this.$nextTick(()=> {
                    window.open(result["url"]+"?download=1");
                })
            });
        },
    },
    mounted() {
        // Load node info
        axios
            .get(`/api/projects/${this.project_id}`)
            .then(response => {
                this.project = response.data;
                for(var prop in this.project){
                    this.items.push({'Category':prop,'Info':this.project[prop]})
                }
                
                console.log(response.data);

                EventBus.$emit("set-title", this.project.name);
            })
            .catch(e => {
                console.log(e);
            });
    },
};
</script>

<style>
#project {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}
</style>
