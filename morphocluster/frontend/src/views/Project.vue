<template>
    <div id="Project">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark text-light">
            <router-link class="navbar-brand text-light" to="/"
                >MorphoCluster</router-link
            >
            <ul class="navbar-nav mr-auto">
                <li class="nav-item active text-light">Projects</li>
            </ul>
            <dark-mode-control />
        </nav>

        <table>
            
            <tbody>
                <tr>
                    <td> Created on </td>
                    <td> {{ project.creation_date }}</td> 
                </tr>
                <tr>
                    <td> Name </td>
                    <td> {{ project.name }}</td>
                </tr>
                <tr>
                    <td> Node_id  </td>
                    <td> {{ project.node_id }}</td>
                </tr>
                <tr> 
                    <td> Project_id </td>
                    <td> {{ project.project_id }}</td>
                </tr>
                <tr>
                    <td> Visible  </td>
                    <td> {{ project.visible }}</td> 
                </tr>    
                    
            </tbody>
        </table> 
    </div>
</template>

<script>
import axios from "axios";
import { EventBus } from "@/event-bus.js";
import DarkModeControl from "@/components/DarkModeControl.vue";


export default {
    name: "ProjectView",
    props: {"project_id": Number},
    components: {DarkModeControl},
    data() {
        return {
            project: null,
        };
    },
    methods: {

    },
    mounted() {4
        // Load node info
        axios
            .get(`/api/projects/${this.project_id}`)
            .then(response => {
                this.project = response.data;
                console.log(response.data);

                EventBus.$emit("set-title", this.project.name);
            })
            .catch(e => {
                console.log(e);
            });
    }
};
</script>

<style>
#Project {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}
table {
    
    width: 100%;
}

th, td {
  padding: 8px;
  border-bottom: 1px solid #ddd;
}

th {
  background-color: #f2f2f2;
}
</style>
