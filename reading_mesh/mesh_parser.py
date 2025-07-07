import gmsh as gm
import numpy as np
import matplotlib.pyplot as plt 
from reading_mesh.Mesh_c import *
from solver_function.function_fem import field_fem 
from solver_function.function_fem import element_type as ET
from solver_function.bc import extract_node_internal_BC
from reading_mesh.Mesh_c import _find_boundary_edge



dict_boundary = {101:'Subduction_surface',
     102:'Channel_surface',
     104:'OVP_v0',
     105:'OVP_v1',
     111:'top',
     112:'right',
     113:'bottom',
     114:'left'}
dict_eT = {'t_P_2_7':'Second order triangle, with barycentral node P_2^+',
           't_P_2_6':'Second order triangle P_2',
           't_P_1_3':'Triangle with discontinous element P_-1'}

def _create_computational_mesh(fname:str,ndim:int,mV:int,mP:int,mT:int):
    """
    function to read the mesh data and a central node and update the node data 
    -> This function needs to 
    a. Adding a node per each element
    b. Recomputing the index of the node, such that everything is consistent 
    c. Provide a connectivity array {which I do not know how to do}
    ============
    1. Read the data of the mesh
    A. Find Element & Node 
    B. Find Per each physical line their node and save
    C.  = Add a new node per each element (and computing the coordinate)
    D.  = Update the other array
    E.  = > Finalise the mesh 
    ============ 
    """
    # Initialise gm 
    gm.initialize()
    gm.open(fname)
    # Read the node
    # get the number of nodes
    node_num,coord,param = gm.model.mesh.getNodes(-1,-1)
    # create x,y arrays
    x = np.zeros((len(node_num),1),dtype=float)
    y = np.zeros((len(node_num),1),dtype=float)
    # extract the coordinate
    xyz=coord.reshape([len(node_num),3])
    x = xyz[:,0]
    y = xyz[:,1]    
    # get the number of elements    
    element_type = gm.model.mesh.getElementTypes()
    
    if (len(element_type)>2) or ((8 in element_type) == False) or ((9 in element_type) == False):
        raise ValueError("The mesh that you created is not of the right type. Please create a mesh with second order triangles.")
    
    EL,EL_nodes = gm.model.mesh.getElementsByType(9,-1)

    num_elements = len(EL)
    EL_nodes = EL_nodes.reshape([-1,6])
    #define_new_nodes(x,y)    
    # get the number of physical groups
    npg = gm.model.getPhysicalGroups(1)
    # get the number of nodes in the physical group 1

    # Extract Boundary & internal boundary
    list_boundary = []
    for i in range(len(npg)):
        if npg[i][0] == 1:
            node_B=extract_node_boundary(npg,i)
        list_boundary.append([dict_boundary[npg[i][1]],node_B])

    EL,EL_nodes,num_new,xnew,ynew=add_barycenter_node(EL,EL_nodes,node_num,x,y)

    gm.finalize()
    EL = np.array(EL,dtype=np.int32)
    EL_nodes = np.array(EL_nodes,dtype=np.int32)
    xnew = np.array(xnew,dtype=np.float64)
    ynew = np.array(ynew,dtype=np.float64)
    num_new = np.array(num_new,dtype=np.int32)
    
    # fill the mesh object 
    mesh = MESH(num_new, xnew, ynew, EL, EL_nodes, len(EL), len(num_new))
    
    CD = Computational_Data(ndim=2,mV=7,mP=3,mT=6,elementV=ET.t_P_2_7.value,elementT=ET.t_P_2_6.value,elementP=ET.t_P_1_3.value,nel=mesh.nel)
    
    CD._update_(mesh.nv,mesh.nel)
    
    bc_int = extract_node_internal_BC(list_boundary,mesh.nv)

#    mesh = _find_boundary_edge(mesh, CD,'right')
#    mesh = _find_boundary_edge(mesh, CD,'left')
#    mesh = _find_boundary_edge(mesh, CD,'bottom')

    
    print('===================================Triangular mesh of: ')
    print('=====Lx [total domain]= [%.2f,%2f] [km]'%(np.min(xnew),np.max(xnew)))
    print('=====Ly [total domain]= [%.2f,%2f] [km]'%(np.min(ynew),np.max(ynew)))
    print('=====number of node   = [%d]         []'%(mesh.nv))
    print('=====number of element= [%d]         []'%(mesh.nel))    
    print(f'=====element velocity: {dict_eT[ET.t_P_2_7.name]}') 
    print(f'=====element pressure: {dict_eT[ET.t_P_1_3.name]}') 
    print(f'=====element temperature: {dict_eT[ET.t_P_2_6.name]}') 
    print('======================================================= ')



    return mesh,CD,bc_int 

def extract_node_boundary(npg,i): 
    """
    Small function to extract the boundary node for any given physical line
    existing in the model domain
    Input:
    npg: group data
    i  : nth group
    Output: 
    nodes_B: node of the given physical group
    """
    
    n_bound=[]
    print(dict_boundary[npg[i][1]])
    n_pg = gm.model.getEntitiesForPhysicalGroup(npg[i][0],npg[i][1])
    for ig in n_pg: 
        cc = gm.model.mesh.getElementsByType(8,int(ig))
        for ic in cc[1]:
            n_bound.append(int(ic))
        nodes_B = np.unique(n_bound)
    return nodes_B

def add_barycenter_node(el,eln,nnum,x,y):
    """
    Input: 
    el: element number
    eln:  connectivity element 
    nnum: number of node
    x: coordinate x 
    y: coordinate y 
    =====
    Output: 
    el: corrected numbering of element {gmsh numerate the triangular element after the line element}
    elnn: updated connectivity matrix
    nnumn : update node list
    xn : update coordinate x
    yn : update coordinate y 
    """
    # Correcting element numbering
    # The line that defines the internal boundaries are considered elements, and the node are numbered as a function of them. The line element need to be purged: 
    el = el - np.min(el)+1
    num_new_node = len(el)+len(nnum)
    xn = np.zeros([(num_new_node)],dtype=float)
    yn = np.zeros([(num_new_node)],dtype=float)
    num_new = np.zeros([num_new_node],dtype=float)
    num_new = np.linspace(1,num_new_node,num=num_new_node).astype('int')

    elnn = np.zeros([len(el),7],dtype=int)
    xn[0:len(nnum)]=x
    yn[0:len(nnum)]=y

    for i in range(len(el)):
       enode = i + len(nnum)+1
       a= [x[eln[i,0]-1],x[eln[i,1]-1],x[eln[i,2]-1]]
       b= [y[eln[i,0]-1],y[eln[i,1]-1],y[eln[i,2]-1]]
       cx = (np.sum(a))/3
       cy = (np.sum(b))/3
       elnn[i,0:6] = eln[i,:]
       elnn[i,6]=num_new[enode-1]
       xn[enode-1]=cx 
       yn[enode-1]=cy                 




    return el,elnn,num_new,xn,yn




     
