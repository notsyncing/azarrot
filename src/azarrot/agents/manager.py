import dataclasses
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import dataclass_wizard
from sqlakeyset import select_page
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from azarrot.agents.common_data import AgentToolRequest
from azarrot.agents.utils import convert_agent_tool_request_to_database
from azarrot.common_data import PageResult
from azarrot.database_schemas import Agent, AgentTool
from azarrot.utils import sanitize_uuid


@dataclass
class AgentGenerationParameters:
    temperature: float = 1
    top_p: float = 1


@dataclass
class AgentInfo:
    id: str
    name: str | None
    description: str | None
    model_id: str
    model_instruction: str | None
    default_generation_parameters: AgentGenerationParameters | None
    additional_data: str | None
    create_time: datetime
    update_time: datetime

    @classmethod
    def from_db_agent(cls, agent: Agent) -> "AgentInfo":
        if agent.default_generation_parameters is not None:
            gen_params = dataclass_wizard.fromdict(
                AgentGenerationParameters, json.loads(agent.default_generation_parameters)
            )
        else:
            gen_params = None

        return cls(
            id=str(agent.id),
            name=agent.name,
            description=agent.description,
            model_id=agent.model_id,
            model_instruction=agent.model_instruction,
            default_generation_parameters=gen_params,
            additional_data=agent.additional_data,
            create_time=agent.create_time,
            update_time=agent.update_time,
        )


@dataclass
class AgentToolInfo:
    agent_id: str
    tool_name: str
    is_internal_tool: bool
    tool_preset_parameters: dict[str, Any] | None
    create_time: datetime
    update_time: datetime

    @classmethod
    def from_db_agent_tool(cls, agent_tool: AgentTool) -> "AgentToolInfo":
        return cls(
            agent_id=str(agent_tool.agent_id),
            tool_name=agent_tool.tool_name,
            is_internal_tool=agent_tool.is_internal_tool,
            tool_preset_parameters=json.loads(agent_tool.tool_preset_parameters or "{}"),
            create_time=agent_tool.create_time,
            update_time=agent_tool.update_time,
        )


@dataclass
class AgentListPagedQuery:
    create_time_desc_order: bool = False
    page_size: int = 20
    before_id: str | uuid.UUID | None = None
    after_id: str | uuid.UUID | None = None


class AgentManager:
    _database: Engine

    def __init__(self, database: Engine) -> None:
        self._database = database

    def create(
        self,
        model_id: str,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        default_generation_parameters: AgentGenerationParameters | None = None,
        tools: list[AgentToolRequest] | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> AgentInfo:
        agent_id = uuid.uuid4()
        now = datetime.now()

        default_gen_param_text = None

        if default_generation_parameters is not None:
            default_gen_param_text = json.dumps(dataclasses.asdict(default_generation_parameters))

        with Session(self._database) as db:
            agent = Agent(
                id=agent_id,
                name=name,
                description=description,
                model_id=model_id,
                model_instruction=instructions,
                default_generation_parameters=default_gen_param_text,
                additional_data=json.dumps(additional_data) if additional_data is not None else None,
                create_time=now,
                update_time=now,
            )

            db.add(agent)

            agent_tools = []

            if tools is not None:
                agent_tools = convert_agent_tool_request_to_database(agent_id, tools, now, now)
                db.add_all(agent_tools)

            db.commit()

            return AgentInfo.from_db_agent(agent)

    def get_enabled_tools(self, agent_id: str | uuid.UUID) -> list[AgentToolInfo]:
        agent_id = sanitize_uuid(agent_id)

        with Session(self._database) as db:
            agent_tools = db.query(AgentTool).filter(AgentTool.agent_id == agent_id).all()

            return [AgentToolInfo.from_db_agent_tool(tool) for tool in agent_tools]

    def get_list(self, query: AgentListPagedQuery) -> PageResult[AgentInfo]:
        if query.before_id is not None and query.after_id is not None:
            raise ValueError("You cannot specify both before_id and after_id!")

        page_border_id = None

        if query.before_id is not None:
            page_border_id = sanitize_uuid(query.before_id)
        elif query.after_id is not None:
            page_border_id = sanitize_uuid(query.after_id)

        with Session(self._database) as db_session:
            page_border_keyset = None

            if page_border_id is not None:
                page_border_create_time = db_session.execute(
                    select(Agent.create_time).where(Agent.id == page_border_id)
                ).scalar_one_or_none()

                if page_border_create_time is None:
                    raise ValueError(f"Specified page border item id {page_border_id} does not exist!")

                page_border_keyset = (page_border_create_time, page_border_id)

            q = select(Agent)

            if query.create_time_desc_order:
                q = q.order_by(Agent.create_time.desc(), Agent.id.desc())
            else:
                q = q.order_by(Agent.create_time, Agent.id)

            before = page_border_keyset if query.before_id is not None else None
            after = page_border_keyset if query.after_id is not None else None

            data = select_page(db_session, q, per_page=query.page_size, before=before, after=after)

            return PageResult(
                data=[AgentInfo.from_db_agent(r._tuple()[0]) for r in data],  # noqa: SLF001
                is_last_page=not data.paging.has_next,
            )

    def get(self, agent_id: str | uuid.UUID) -> AgentInfo | None:
        agent_id = sanitize_uuid(agent_id)

        with Session(self._database) as db:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()

            if agent is None:
                return None

            return AgentInfo.from_db_agent(agent)

    def update(
        self,
        agent_id: str | uuid.UUID,
        model_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        default_generation_parameters: AgentGenerationParameters | None = None,
        tools: list[AgentToolRequest] | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> AgentInfo:
        agent_id = sanitize_uuid(agent_id)
        now = datetime.now()

        with Session(self._database) as db:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()

            if agent is None:
                raise ValueError(f"Agent {agent_id} does not exist!")

            if model_id is not None:
                agent.model_id = model_id

            if name is not None:
                agent.name = name

            if description is not None:
                agent.description = description

            if instructions is not None:
                agent.model_instruction = instructions

            if default_generation_parameters is not None:
                if agent.default_generation_parameters is not None:
                    original_gen_params = dataclass_wizard.fromdict(
                        AgentGenerationParameters, json.loads(agent.default_generation_parameters)
                    )

                    if default_generation_parameters.temperature is not None:
                        original_gen_params.temperature = default_generation_parameters.temperature

                    if default_generation_parameters.top_p is not None:
                        original_gen_params.top_p = default_generation_parameters.top_p

                    agent.default_generation_parameters = json.dumps(dataclass_wizard.asdict(original_gen_params))
                else:
                    agent.default_generation_parameters = json.dumps(
                        dataclass_wizard.asdict(default_generation_parameters)
                    )

            if tools is not None:
                db.execute(delete(AgentTool).where(AgentTool.agent_id == agent.id))

                agent_tools = convert_agent_tool_request_to_database(agent.id, tools, now, now)
                db.add_all(agent_tools)

            if additional_data is not None:
                agent.additional_data = json.dumps(additional_data)

            agent.update_time = now
            db.commit()

            return AgentInfo.from_db_agent(agent)

    def delete(self, agent_id: str | uuid.UUID) -> None:
        agent_id = sanitize_uuid(agent_id)

        with Session(self._database) as db:
            db.execute(delete(AgentTool).where(AgentTool.agent_id == agent_id))

            r = db.execute(delete(Agent).where(Agent.id == agent_id))

            if r.rowcount <= 0:
                raise ValueError(f"Agent {agent_id} does not exist or failed to delete!")

            db.commit()

    def clear_database(self) -> None:
        with Session(self._database) as db:
            db.execute(delete(AgentTool))
            db.execute(delete(Agent))
            db.commit()
